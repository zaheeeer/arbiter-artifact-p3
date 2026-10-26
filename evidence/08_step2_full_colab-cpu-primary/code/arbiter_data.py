"""Public-data acquisition and DuckDB workload preparation for Colab."""

from __future__ import annotations

import hashlib
import gzip
import json
import math
from pathlib import Path
import re
import shutil
import urllib.request
import zipfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    import duckdb
except Exception:  # pragma: no cover - notebooks install the pinned dependency.
    duckdb = None


NYC_TAXI_URL = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/"
    "yellow_tripdata_2024-01.parquet"
)
UCI_ADULT_URL = "https://archive.ics.uci.edu/static/public/2/adult.zip"
NYC_TAXI_SHA256 = "c4d59da7bbc8abaeeeb1727947ee93d9891a71acb42854bd80db1571b2030510"
UCI_ADULT_SHA256 = "7537312dd56c2b98035880805ce99e68183a30ee468aa5329d6df0fbb3cc21bb"
NYC_TAXI_ROWS = 2_964_624
UCI_ADULT_ROWS = 48_842

# DuckDB's deterministic TPC-H generator cardinalities for the three suite
# modes.  Pinning every table prevents a partial or differently scaled cached
# database from being accepted merely because ``orders`` has the right size.
TPCH_EXPECTED_COUNTS: dict[float, dict[str, int]] = {
    0.1: {
        "lineitem": 600_572,
        "orders": 150_000,
        "customer": 15_000,
        "part": 20_000,
        "supplier": 1_000,
        "partsupp": 80_000,
        "nation": 25,
        "region": 5,
    },
    1.0: {
        "lineitem": 6_001_215,
        "orders": 1_500_000,
        "customer": 150_000,
        "part": 200_000,
        "supplier": 10_000,
        "partsupp": 800_000,
        "nation": 25,
        "region": 5,
    },
    10.0: {
        "lineitem": 59_986_052,
        "orders": 15_000_000,
        "customer": 1_500_000,
        "part": 2_000_000,
        "supplier": 100_000,
        "partsupp": 8_000_000,
        "nation": 25,
        "region": 5,
    },
}


def _expected_tpch_counts(scale_factor: float) -> dict[str, int]:
    for supported_scale, counts in TPCH_EXPECTED_COUNTS.items():
        if math.isclose(
            float(scale_factor),
            supported_scale,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            return dict(counts)
    supported = ", ".join(format(value, "g") for value in TPCH_EXPECTED_COUNTS)
    raise ValueError(
        "Unsupported TPC-H scale factor for this evidence suite: "
        f"{scale_factor}. Supported values are {supported}."
    )


def _observed_tpch_counts(connection) -> dict[str, int] | None:
    tables = list(next(iter(TPCH_EXPECTED_COUNTS.values())).keys())
    present = int(
        connection.execute(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_schema = current_schema() AND table_name IN "
            "('lineitem','orders','customer','part','supplier',"
            "'partsupp','nation','region')"
        ).fetchone()[0]
    )
    if present != len(tables):
        return None
    return {
        table: int(
            connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        )
        for table in tables
    }


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_if_missing(url: str, destination: str | Path) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return path
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.unlink(missing_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Arbiter-ICDE-evidence/2.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            with temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
        if temporary.stat().st_size <= 0:
            raise RuntimeError(f"downloaded input is empty: {url}")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return path


def _validate_input_hash(path: Path, expected_sha256: str, label: str) -> str:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"{label} input is missing or empty: {path}")
    observed = file_sha256(path)
    if observed != expected_sha256:
        raise ValueError(
            f"{label} SHA-256 mismatch. Expected {expected_sha256}, observed "
            f"{observed}. Delete the cached file and rerun; do not substitute "
            "a different release silently."
        )
    return observed


def _validate_nyc_parquet(path: Path) -> str:
    observed = _validate_input_hash(path, NYC_TAXI_SHA256, "NYC Taxi")
    with path.open("rb") as handle:
        if handle.read(4) != b"PAR1":
            raise ValueError("NYC Taxi input does not have a Parquet header")
        handle.seek(-4, 2)
        if handle.read(4) != b"PAR1":
            raise ValueError("NYC Taxi input does not have a Parquet footer")
    return observed


def _validate_adult_zip(path: Path) -> str:
    observed = _validate_input_hash(path, UCI_ADULT_SHA256, "UCI Adult")
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            required = {"adult.data", "adult.test"}
            if not required <= names:
                raise ValueError(
                    "UCI Adult archive lacks required files: "
                    + ", ".join(sorted(required - names))
                )
            bad_member = archive.testzip()
            if bad_member is not None:
                raise ValueError(f"UCI Adult archive member is corrupt: {bad_member}")
    except zipfile.BadZipFile as exc:
        raise ValueError("UCI Adult input is not a valid ZIP archive") from exc
    return observed


ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]


def load_adult_frame(zip_path: str | Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as archive:
        train = pd.read_csv(
            archive.open("adult.data"),
            names=ADULT_COLUMNS,
            skipinitialspace=True,
            na_values="?",
        )
        test = pd.read_csv(
            archive.open("adult.test"),
            names=ADULT_COLUMNS,
            skipinitialspace=True,
            na_values="?",
            skiprows=1,
        )
    frame = pd.concat([train, test], ignore_index=True)
    frame["income"] = frame["income"].astype("string").str.rstrip(".")
    return frame


def _require_duckdb():
    if duckdb is None:
        raise RuntimeError("duckdb is required; run the notebook installation cell")


def _load_tpch_extension(connection, extension_directory: Path) -> str:
    """Load TPC-H without silently replacing it when repository access fails."""

    try:
        connection.execute("LOAD tpch")
        return "cached"
    except Exception:
        pass

    repository_error: Exception | None = None
    try:
        connection.execute("INSTALL tpch")
        connection.execute("LOAD tpch")
        return "duckdb_core_repository"
    except Exception as exc:
        repository_error = exc

    version = str(connection.execute("SELECT version()").fetchone()[0])
    platform_name = str(connection.execute("PRAGMA platform").fetchone()[0])
    url = (
        f"https://extensions.duckdb.org/{version}/{platform_name}/"
        "tpch.duckdb_extension.gz"
    )
    install_directory = extension_directory / version / platform_name
    install_directory.mkdir(parents=True, exist_ok=True)
    target = install_directory / "tpch.duckdb_extension"
    compressed_part = target.with_suffix(target.suffix + ".gz.part")
    extension_part = target.with_suffix(target.suffix + ".part")
    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Arbiter-ICDE-evidence/1.0"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            with compressed_part.open("wb") as output:
                shutil.copyfileobj(response, output)
        with gzip.open(compressed_part, "rb") as source:
            with extension_part.open("wb") as output:
                shutil.copyfileobj(source, output)
        if extension_part.stat().st_size <= 0:
            raise RuntimeError("downloaded TPC-H extension is empty")
        extension_part.replace(target)
        compressed_part.unlink(missing_ok=True)
        connection.execute("LOAD tpch")
        return "duckdb_https_repository_fallback"
    except Exception as fallback_error:
        compressed_part.unlink(missing_ok=True)
        extension_part.unlink(missing_ok=True)
        raise RuntimeError(
            "TPC-H extension installation failed through both DuckDB's core "
            f"repository and the HTTPS fallback. Core error: {repository_error}; "
            f"HTTPS error: {fallback_error}"
        ) from fallback_error


def prepare_duckdb(
    *,
    project_root: str | Path,
    tpch_scale_factor: float,
    force_rebuild: bool = False,
    nyc_parquet_path: str | Path | None = None,
    adult_zip_path: str | Path | None = None,
):
    """Download public inputs, build TPC-H, and return a persistent connection."""

    _require_duckdb()
    scale_factor = float(tpch_scale_factor)
    if not math.isfinite(scale_factor) or scale_factor <= 0.0:
        raise ValueError("tpch_scale_factor must be finite and strictly positive")
    expected_tpch_counts = _expected_tpch_counts(scale_factor)
    root = Path(project_root)
    inputs = root / "inputs"
    state = root / "state"
    inputs.mkdir(parents=True, exist_ok=True)
    state.mkdir(parents=True, exist_ok=True)

    nyc_path = (
        Path(nyc_parquet_path)
        if nyc_parquet_path
        else download_if_missing(
            NYC_TAXI_URL, inputs / "yellow_tripdata_2024-01.parquet"
        )
    )
    adult_path = (
        Path(adult_zip_path)
        if adult_zip_path
        else download_if_missing(UCI_ADULT_URL, inputs / "adult.zip")
    )
    nyc_sha256 = _validate_nyc_parquet(nyc_path)
    adult_sha256 = _validate_adult_zip(adult_path)

    database_path = state / f"arbiter_sf{scale_factor:g}.duckdb"
    if force_rebuild and database_path.exists():
        database_path.unlink()
    connection = duckdb.connect(str(database_path))
    extension_directory = state / "duckdb_extensions"
    extension_directory.mkdir(parents=True, exist_ok=True)
    escaped_extension_directory = extension_directory.as_posix().replace(
        "'", "''"
    )
    connection.execute(
        f"SET extension_directory='{escaped_extension_directory}'"
    )

    escaped_nyc_path = nyc_path.resolve().as_posix().replace("'", "''")
    connection.execute(
        f"""
        CREATE OR REPLACE VIEW nyc_taxi AS
        SELECT * FROM read_parquet('{escaped_nyc_path}')
        """
    )

    nyc_columns = {
        str(row[0])
        for row in connection.execute("DESCRIBE nyc_taxi").fetchall()
    }
    required_nyc_columns = {
        "VendorID",
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "trip_distance",
        "fare_amount",
        "total_amount",
    }
    if not required_nyc_columns <= nyc_columns:
        raise ValueError(
            "NYC Taxi schema mismatch; missing columns: "
            + ", ".join(sorted(required_nyc_columns - nyc_columns))
        )
    nyc_rows = int(connection.execute("SELECT COUNT(*) FROM nyc_taxi").fetchone()[0])
    if nyc_rows != NYC_TAXI_ROWS:
        raise ValueError(
            f"NYC Taxi row-count mismatch: expected {NYC_TAXI_ROWS}, observed {nyc_rows}"
        )

    adult = load_adult_frame(adult_path)
    if len(adult) != UCI_ADULT_ROWS or list(adult.columns) != ADULT_COLUMNS:
        raise ValueError(
            "UCI Adult content/schema mismatch: expected "
            f"{UCI_ADULT_ROWS} rows and the canonical 15 columns"
        )
    connection.register("adult_input_frame", adult)
    connection.execute(
        "CREATE OR REPLACE TABLE uci_adult AS SELECT * FROM adult_input_frame"
    )
    connection.unregister("adult_input_frame")

    tpch_tables = list(expected_tpch_counts)
    observed_tpch_counts: dict[str, int] | None = None
    try:
        observed_tpch_counts = _observed_tpch_counts(connection)
    except Exception:
        observed_tpch_counts = None
    if observed_tpch_counts != expected_tpch_counts:
        for table in tpch_tables:
            connection.execute(f'DROP TABLE IF EXISTS "{table}"')
        tpch_extension_source = _load_tpch_extension(
            connection, extension_directory
        )
        connection.execute(f"CALL dbgen(sf={scale_factor})")
        observed_tpch_counts = _observed_tpch_counts(connection)
        if observed_tpch_counts != expected_tpch_counts:
            raise RuntimeError(
                "TPC-H materialization failed exact cardinality validation. "
                f"Expected {expected_tpch_counts}; observed "
                f"{observed_tpch_counts}."
            )
    else:
        tpch_extension_source = "existing_tables"

    provenance = pd.DataFrame(
        [
            {
                "dataset": "NYC Taxi 2024-01",
                "provenance": "public observational",
                "source_url": NYC_TAXI_URL,
                "local_file": nyc_path.as_posix(),
                "sha256": nyc_sha256,
                "total_rows": nyc_rows,
            },
            {
                "dataset": f"TPC-H SF{scale_factor:g}",
                "provenance": "benchmark-generated",
                "source_url": "https://www.tpc.org/tpch/",
                "local_file": database_path.as_posix(),
                "sha256": "",
                "materialization": tpch_extension_source,
                "total_rows": int(sum(observed_tpch_counts.values())),
            },
            {
                "dataset": "UCI Adult",
                "provenance": "public observational",
                "source_url": UCI_ADULT_URL,
                "local_file": adult_path.as_posix(),
                "sha256": adult_sha256,
                "total_rows": int(
                    connection.execute("SELECT COUNT(*) FROM uci_adult").fetchone()[0]
                ),
            },
        ]
    )
    return connection, provenance


def _numeric_boundaries(
    connection,
    table: str,
    column: str,
    quantiles: Sequence[float],
) -> list[float]:
    quantile_sql = ", ".join(format(float(value), ".8g") for value in quantiles)
    query = (
        f'SELECT quantile_cont("{column}", [{quantile_sql}]) '
        f'FROM "{table}" WHERE "{column}" IS NOT NULL '
        f'AND isfinite(CAST("{column}" AS DOUBLE))'
    )
    values = connection.execute(query).fetchone()[0]
    finite_values: set[float] = set()
    for value in values:
        if value is None:
            continue
        # DuckDB returns DECIMAL quantiles as ``decimal.Decimal``. NumPy's
        # ``isfinite`` does not accept that type on all Colab NumPy versions, so
        # normalize first and validate the converted scalar.
        converted = float(value)
        if math.isfinite(converted):
            finite_values.add(converted)
    return sorted(finite_values)


def _top_categories(
    connection,
    table: str,
    column: str,
    limit: int,
) -> list[str]:
    query = (
        f'SELECT CAST("{column}" AS VARCHAR) AS value, COUNT(*) AS frequency '
        f'FROM "{table}" WHERE "{column}" IS NOT NULL '
        f'GROUP BY 1 ORDER BY frequency DESC, value ASC LIMIT {int(limit)}'
    )
    return [str(row[0]) for row in connection.execute(query).fetchall()]


def collect_table_spec(
    connection,
    *,
    table: str,
    output_columns: Sequence[str],
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    measure_columns: Sequence[str],
    quantile_count: int,
    category_limit: int = 12,
) -> dict[str, Any]:
    quantiles = np.linspace(0.0, 1.0, int(quantile_count)).tolist()
    return {
        "table": table,
        "row_count": int(
            connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        ),
        "output_columns": list(output_columns),
        "numeric_columns": {
            column: _numeric_boundaries(connection, table, column, quantiles)
            for column in numeric_columns
        },
        "categorical_columns": {
            column: _top_categories(connection, table, column, category_limit)
            for column in categorical_columns
        },
        "measure_columns": list(measure_columns),
    }


def default_dataset_specs(
    connection,
    *,
    tpch_scale_factor: float,
    quantile_count: int = 25,
) -> dict[str, list[dict[str, Any]]]:
    return {
        "NYC Taxi 2024-01": [
            collect_table_spec(
                connection,
                table="nyc_taxi",
                output_columns=[
                    "VendorID",
                    "tpep_pickup_datetime",
                    "tpep_dropoff_datetime",
                    "passenger_count",
                    "trip_distance",
                    "RatecodeID",
                    "PULocationID",
                    "DOLocationID",
                    "payment_type",
                    "fare_amount",
                    "total_amount",
                ],
                numeric_columns=[
                    "trip_distance",
                    "fare_amount",
                    "total_amount",
                    "passenger_count",
                ],
                categorical_columns=[
                    "VendorID",
                    "RatecodeID",
                    "payment_type",
                ],
                measure_columns=["fare_amount", "total_amount", "trip_distance"],
                quantile_count=quantile_count,
            )
        ],
        f"TPC-H SF{tpch_scale_factor:g}": [
            collect_table_spec(
                connection,
                table="lineitem",
                output_columns=[
                    "l_orderkey",
                    "l_partkey",
                    "l_suppkey",
                    "l_linenumber",
                    "l_quantity",
                    "l_extendedprice",
                    "l_discount",
                    "l_shipdate",
                    "l_returnflag",
                    "l_linestatus",
                ],
                numeric_columns=[
                    "l_quantity",
                    "l_extendedprice",
                    "l_discount",
                    "l_tax",
                ],
                categorical_columns=["l_returnflag", "l_linestatus", "l_shipmode"],
                measure_columns=["l_quantity", "l_extendedprice", "l_discount"],
                quantile_count=quantile_count,
            ),
            collect_table_spec(
                connection,
                table="orders",
                output_columns=[
                    "o_orderkey",
                    "o_custkey",
                    "o_orderstatus",
                    "o_totalprice",
                    "o_orderdate",
                    "o_orderpriority",
                    "o_shippriority",
                ],
                numeric_columns=["o_totalprice", "o_shippriority"],
                categorical_columns=["o_orderstatus", "o_orderpriority"],
                measure_columns=["o_totalprice"],
                quantile_count=quantile_count,
            ),
            collect_table_spec(
                connection,
                table="customer",
                output_columns=[
                    "c_custkey",
                    "c_name",
                    "c_address",
                    "c_nationkey",
                    "c_phone",
                    "c_acctbal",
                    "c_mktsegment",
                ],
                numeric_columns=["c_acctbal", "c_nationkey"],
                categorical_columns=["c_mktsegment", "c_nationkey"],
                measure_columns=["c_acctbal"],
                quantile_count=quantile_count,
            ),
        ],
        "UCI Adult": [
            collect_table_spec(
                connection,
                table="uci_adult",
                output_columns=[
                    "age",
                    "workclass",
                    "education",
                    "education_num",
                    "marital_status",
                    "occupation",
                    "relationship",
                    "race",
                    "sex",
                    "hours_per_week",
                    "native_country",
                    "income",
                ],
                numeric_columns=[
                    "age",
                    "education_num",
                    "capital_gain",
                    "capital_loss",
                    "hours_per_week",
                ],
                categorical_columns=[
                    "workclass",
                    "education",
                    "occupation",
                    "relationship",
                    "race",
                    "sex",
                    "income",
                ],
                measure_columns=[
                    "age",
                    "education_num",
                    "capital_gain",
                    "hours_per_week",
                ],
                quantile_count=quantile_count,
            )
        ],
    }


ESTIMATED_ROWS_PATTERN = re.compile(r"~([\d,]+(?:\.\d+)?) rows")


def extract_plan_features(connection, assets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    operators = [
        "SEQ_SCAN",
        "TABLE_SCAN",
        "FILTER",
        "PROJECTION",
        "HASH_JOIN",
        "NESTED_LOOP_JOIN",
        "GROUP_BY",
        "AGGREGATE",
        "ORDER_BY",
        "LIMIT",
    ]
    for row in assets.itertuples(index=False):
        started = __import__("time").perf_counter_ns()
        try:
            plan_rows = connection.execute(f"EXPLAIN {row.sql}").fetchall()
            plan = "\n".join(" ".join(str(value) for value in values) for values in plan_rows)
            estimates = [
                float(value.replace(",", ""))
                for value in ESTIMATED_ROWS_PATTERN.findall(plan)
            ]
            status = "ok"
            error = ""
        except Exception as exc:
            plan = ""
            estimates = []
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
        record = {
            "dataset": row.dataset,
            "asset_id": row.asset_id,
            "plan_status": status,
            "plan_error": error,
            "estimated_rows": max(estimates, default=0.0),
            "plan_chars": len(plan),
            "explain_ms": (
                __import__("time").perf_counter_ns() - started
            )
            / 1e6,
        }
        upper_plan = plan.upper()
        for operator in operators:
            record[f"plan_{operator.lower()}"] = upper_plan.count(operator)
        rows.append(record)
    return pd.DataFrame(rows)


def save_json(path: str | Path, value: Mapping[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
