"""Targeted performance hardening for the ARBITER closure campaign.

This module is deliberately downstream of two immutable evidence layers:

* audited-v4 Notebooks 00--04 provide the supported SQL assets, compiled
  constraints, controlled prices, and system-level evidence;
* request-bound closure v1.1 (Notebook 05) provides the unchanged signed-token
  protocol and its correctness evidence.

Notebook 06 corrects a benchmark contract that staged an answer explicitly and
then staged it again inside ``RequestBoundSnapshotStore.issue_token``.  The
primary path below calls ``issue_token`` exactly once and instruments its one
internal durable stage.  The signed token format and release implementation are
not modified.

The fallback experiment also avoids a Python loop over satisfied constraint
rows.  It reuses the audited active cyclic kernel with a budget equal to the
entire current violation set, retains the independent verifier after every
tier, and retains OSQP followed by rejection as the fail-closed backstop.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import secrets
import threading
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

import arbiter_core as core
from arbiter_request_release import (
    BoundAnswerBuffer,
    DurableReleaseLedger,
    RequestBoundSnapshotStore,
    RequestContext,
    canonical_answer_bytes,
    exact_sql_digest,
)


AUDITED_SUITE_VERSION = "2026-08-01-audited-v4"
CLOSURE_SUITE_VERSION = "2026-08-01-request-bound-closure-v1.1"
HARDENING_SUITE_VERSION = "2026-08-12-performance-hardening-v1.0"

CLOSURE_CODE_HASHES = {
    "arbiter_request_release.py": "b00dc16f4ab5ea9fc57d30fb141c83f31003bc8014e8285c1e1eabd812f66fc3",
    "arbiter_closure_experiments.py": "bf63760ee2124e6976bcf02e17d14c00c083defa4bdedae70fedfdec46e7cfb4",
}


# These are supported audited-v4 assets.  The row bands are preregistered
# experiment categories, not discovered after timing.
ANSWER_PROFILE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "size_class": "scalar",
        "dataset": "TPC-H SF1",
        "asset_id": "asset_2567eb4af28fbdb4",
        "min_rows": 1,
        "max_rows": 1,
        "expected_rows": 1,
    },
    {
        "size_class": "small",
        "dataset": "UCI Adult",
        "asset_id": "asset_ac656b81b6c77c9e",
        "min_rows": 8,
        "max_rows": 255,
        "expected_rows": 21,
    },
    {
        "size_class": "medium",
        "dataset": "UCI Adult",
        "asset_id": "asset_0009e06da018d508",
        "min_rows": 1_024,
        "max_rows": 9_999,
        "expected_rows": 3_267,
    },
    {
        "size_class": "large",
        "dataset": "TPC-H SF1",
        "asset_id": "asset_000f7a1dcff309ce",
        "min_rows": 50_000,
        "max_rows": 200_000,
        "expected_rows": 93_750,
    },
)


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _boolean_series(series: pd.Series, *, name: str) -> pd.Series:
    parsed = series.astype(str).str.lower().map({"true": True, "false": False})
    if parsed.isna().any():
        raise RuntimeError(f"{name} contains a non-boolean value")
    return parsed.astype(bool)


def verify_closure_inputs(closure_dir: Path) -> pd.DataFrame:
    """Fail closed on partial, stale, modified, or non-full Notebook 05 input."""

    closure_dir = closure_dir.resolve()
    required = [
        closure_dir / "closure_run_config.json",
        closure_dir / "checksums.csv",
        closure_dir / "checksums.sha256",
        closure_dir / "stage_quality_checks.csv",
        closure_dir / "input_evidence_audit.csv",
        closure_dir / "claim_audit.csv",
        closure_dir / "code" / "code_manifest.json",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"required closure input is missing: {path}")

    config = json.loads(
        (closure_dir / "closure_run_config.json").read_text(encoding="utf-8")
    )
    if config.get("closure_suite_version") != CLOSURE_SUITE_VERSION:
        raise RuntimeError("Notebook 06 requires request-bound closure v1.1")
    if config.get("closure_mode") != "full":
        raise RuntimeError("Notebook 06 requires a completed full Notebook 05 run")
    if config.get("audited_input_suite_version") != AUDITED_SUITE_VERSION:
        raise RuntimeError("Notebook 05 was not linked to audited-v4 input")
    if config.get("audited_input_run_mode") != "full":
        raise RuntimeError("Notebook 05 was not linked to a full audited-v4 run")

    quality = pd.read_csv(closure_dir / "stage_quality_checks.csv")
    if set(quality.columns) != {"check", "passed"}:
        raise RuntimeError("Notebook 05 quality-check schema changed")
    quality_passed = _boolean_series(quality["passed"], name="Notebook 05 checks")
    if len(quality) != 25 or not quality_passed.all():
        raise RuntimeError("the 25 Notebook 05 quality checks do not all pass")

    prior_input_audit = pd.read_csv(closure_dir / "input_evidence_audit.csv")
    if "passed" not in prior_input_audit.columns or not _boolean_series(
        prior_input_audit["passed"], name="Notebook 05 input audit"
    ).all():
        raise RuntimeError("Notebook 05 did not validate all audited-v4 inputs")

    claims = pd.read_csv(closure_dir / "claim_audit.csv")
    request_claims = claims[
        claims["claim"].astype(str).str.contains(
            "Request-, buyer/session-", regex=False
        )
    ]
    if len(request_claims) != 1 or request_claims.iloc[0]["status"] != "SUPPORTED":
        raise RuntimeError("Notebook 05 request-binding closure is not supported")

    manifest = pd.read_csv(closure_dir / "checksums.csv")
    if set(manifest.columns) != {"path", "bytes", "sha256"}:
        raise RuntimeError("Notebook 05 checksum-manifest schema changed")
    if manifest.empty or manifest["path"].isna().any():
        raise RuntimeError("Notebook 05 checksum manifest is empty or incomplete")
    if manifest["path"].astype(str).duplicated().any():
        raise RuntimeError("Notebook 05 checksum manifest has duplicate paths")
    byte_counts = pd.to_numeric(manifest["bytes"], errors="coerce")
    if byte_counts.isna().any() or (byte_counts < 0).any() or not np.all(
        byte_counts.to_numpy(float) == np.floor(byte_counts.to_numpy(float))
    ):
        raise RuntimeError("Notebook 05 checksum manifest has invalid byte counts")
    if not manifest["sha256"].astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
        raise RuntimeError("Notebook 05 checksum manifest has invalid digests")

    expected_sha_text = "".join(
        f"{row.sha256}  {row.path}\n" for row in manifest.itertuples(index=False)
    )
    if (closure_dir / "checksums.sha256").read_text(
        encoding="utf-8"
    ) != expected_sha_text:
        raise RuntimeError("Notebook 05 CSV and text checksum manifests disagree")

    rows: list[dict[str, Any]] = []
    for record in manifest.itertuples(index=False):
        relative = Path(str(record.path))
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"unsafe Notebook 05 checksum path: {relative}")
        target = (closure_dir / relative).resolve()
        if closure_dir not in target.parents:
            raise RuntimeError(f"Notebook 05 checksum path escaped: {relative}")
        exists = target.is_file()
        observed_bytes = target.stat().st_size if exists else -1
        observed_hash = sha256_file(target) if exists else ""
        passed = bool(
            exists
            and observed_bytes == int(record.bytes)
            and observed_hash == str(record.sha256)
        )
        rows.append(
            {
                "scope": "closure_evidence",
                "path": relative.as_posix(),
                "expected_bytes": int(record.bytes),
                "observed_bytes": int(observed_bytes),
                "expected_sha256": str(record.sha256),
                "observed_sha256": observed_hash,
                "passed": passed,
            }
        )

    code_manifest = json.loads(
        (closure_dir / "code" / "code_manifest.json").read_text(encoding="utf-8")
    )
    if code_manifest != CLOSURE_CODE_HASHES:
        raise RuntimeError("Notebook 05 code manifest is not the audited v1.1 manifest")
    for name, expected_hash in CLOSURE_CODE_HASHES.items():
        target = closure_dir / "code" / name
        observed_hash = sha256_file(target) if target.is_file() else ""
        rows.append(
            {
                "scope": "closure_code",
                "path": f"code/{name}",
                "expected_bytes": None,
                "observed_bytes": target.stat().st_size if target.is_file() else -1,
                "expected_sha256": expected_hash,
                "observed_sha256": observed_hash,
                "passed": observed_hash == expected_hash,
            }
        )

    audit = pd.DataFrame(rows)
    if audit.empty or not audit["passed"].all():
        failed = audit.loc[~audit["passed"], "path"].tolist()
        raise RuntimeError(f"Notebook 05 input integrity failed: {failed[:5]}")
    return audit


def verify_database_and_source_inputs(
    connection: Any,
    *,
    project_root: Path,
    dataset_metadata: pd.DataFrame,
    tpch_scale_factor: float,
) -> pd.DataFrame:
    """Recheck source hashes and every materialized table cardinality.

    The earlier evidence package does not contain a content hash of the DuckDB
    file itself.  This check therefore revalidates the two observational source
    archives by SHA-256 and all ten materialized table counts.  It detects stale,
    partial, differently scaled, or grossly substituted databases; a malicious
    same-cardinality logical rewrite remains outside the available evidence and
    must not be claimed as cryptographic database attestation.
    """

    from arbiter_data import (  # imported lazily from the pinned audited code
        NYC_TAXI_ROWS,
        UCI_ADULT_ROWS,
        _expected_tpch_counts,
    )

    required_metadata = {
        "dataset",
        "local_file",
        "sha256",
        "total_rows",
    }
    if not required_metadata.issubset(dataset_metadata.columns):
        raise ValueError("dataset metadata is missing required fields")
    rows: list[dict[str, Any]] = []
    for dataset in ["NYC Taxi 2024-01", "UCI Adult"]:
        metadata = dataset_metadata[dataset_metadata["dataset"].eq(dataset)]
        if len(metadata) != 1:
            raise RuntimeError(f"dataset metadata is not unique: {dataset}")
        record = metadata.iloc[0]
        expected_hash = str(record["sha256"])
        if len(expected_hash) != 64:
            raise RuntimeError(f"missing source SHA-256 for {dataset}")
        source_path = (project_root / "inputs" / Path(str(record["local_file"])).name).resolve()
        if project_root not in source_path.parents:
            raise RuntimeError(f"source input escaped project root: {source_path}")
        observed_hash = sha256_file(source_path) if source_path.is_file() else ""
        rows.append(
            {
                "scope": "source_file",
                "name": dataset,
                "expected": expected_hash,
                "observed": observed_hash,
                "passed": observed_hash == expected_hash,
            }
        )

    expected_counts = {
        "nyc_taxi": int(NYC_TAXI_ROWS),
        "uci_adult": int(UCI_ADULT_ROWS),
        **{
            str(table): int(count)
            for table, count in _expected_tpch_counts(tpch_scale_factor).items()
        },
    }
    present = {
        str(row[0])
        for row in connection.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = current_schema()"
        ).fetchall()
    }
    for table, expected_count in expected_counts.items():
        exists = table in present
        observed_count = (
            int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
            if exists
            else -1
        )
        rows.append(
            {
                "scope": "duckdb_table_cardinality",
                "name": table,
                "expected": str(expected_count),
                "observed": str(observed_count),
                "passed": exists and observed_count == expected_count,
            }
        )
    audit = pd.DataFrame(rows)
    if len(audit) != 12 or not audit["passed"].all():
        failed = audit.loc[~audit["passed"], "name"].tolist()
        raise RuntimeError(f"database/source integrity failed: {failed[:5]}")
    return audit


class InstrumentedDurableReleaseLedger(DurableReleaseLedger):
    """Measure the one stage call nested inside ``issue_token``.

    The subclass does not alter the transaction, schema, PRAGMAs, payload, or
    return value.  Thread-local counters prevent unrelated worker threads from
    contaminating a timing record.
    """

    def __init__(self, path: str | Path) -> None:
        self._stage_metrics = threading.local()
        super().__init__(path)
        self.reset_stage_metrics()

    def reset_stage_metrics(self) -> None:
        self._stage_metrics.calls = 0
        self._stage_metrics.elapsed_ns = 0

    @property
    def stage_calls(self) -> int:
        return int(getattr(self._stage_metrics, "calls", 0))

    @property
    def stage_elapsed_ns(self) -> int:
        return int(getattr(self._stage_metrics, "elapsed_ns", 0))

    def stage_buffer(self, request: RequestContext, buffer: BoundAnswerBuffer) -> None:
        started = time.perf_counter_ns()
        try:
            super().stage_buffer(request, buffer)
        finally:
            elapsed = time.perf_counter_ns() - started
            self._stage_metrics.calls = self.stage_calls + 1
            self._stage_metrics.elapsed_ns = self.stage_elapsed_ns + int(elapsed)


def execute_actual_answer_profiles(
    assets: pd.DataFrame,
    connection: Any,
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    """Execute four preregistered supported assets against the audited database."""

    required_columns = {"dataset", "asset_id", "sql", "expected_supported"}
    if not required_columns.issubset(assets.columns):
        raise ValueError("SQL asset table is missing required columns")
    profiles: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for spec in ANSWER_PROFILE_SPECS:
        matches = assets[
            assets["dataset"].astype(str).eq(spec["dataset"])
            & assets["asset_id"].astype(str).eq(spec["asset_id"])
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"answer profile asset is not unique: {spec['dataset']}/{spec['asset_id']}"
            )
        asset = matches.iloc[0]
        if str(asset["expected_supported"]).lower() != "true":
            raise RuntimeError(f"answer profile is not supported: {spec['asset_id']}")
        sql = str(asset["sql"])
        started = time.perf_counter_ns()
        answer = connection.execute(sql).fetchall()
        query_ms = (time.perf_counter_ns() - started) / 1e6
        answer_rows = len(answer)
        if not int(spec["min_rows"]) <= answer_rows <= int(spec["max_rows"]):
            raise RuntimeError(
                f"{spec['size_class']} profile returned {answer_rows} rows; "
                f"expected [{spec['min_rows']}, {spec['max_rows']}]"
            )
        if answer_rows != int(spec["expected_rows"]):
            raise RuntimeError(
                f"{spec['size_class']} profile cardinality changed: "
                f"{answer_rows} != {spec['expected_rows']}"
            )
        payload_bytes = canonical_answer_bytes(answer)
        record = {
            **spec,
            "sql": sql,
            "query_digest": exact_sql_digest(sql),
            "sql_sha256": hashlib.sha256(sql.encode("utf-8")).hexdigest(),
            "answer": answer,
            "answer_rows": answer_rows,
            "payload_bytes": len(payload_bytes),
            "query_ms": query_ms,
            "source": "DuckDB execution over audited NYC/Adult/TPC-H database",
        }
        profiles[str(spec["size_class"])] = record
        rows.append({key: value for key, value in record.items() if key not in {"answer", "sql"}})
    if set(profiles) != {"scalar", "small", "medium", "large"}:
        raise RuntimeError("the four answer-size profiles were not constructed")
    return profiles, pd.DataFrame(rows)


def _balanced_profile_order(profile_names: Sequence[str], repetition: int) -> list[str]:
    names = list(profile_names)
    if not names:
        return []
    rotation = int(repetition) % len(names)
    order = names[rotation:] + names[:rotation]
    if (int(repetition) // len(names)) % 2:
        order = list(reversed(order))
    return order


def _stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _bootstrap_median_ci(
    values: Sequence[float],
    *,
    resamples: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError("bootstrap input must be a nonempty finite vector")
    if int(resamples) < 100:
        raise ValueError("bootstrap requires at least 100 resamples")
    generator = np.random.default_rng(int(seed))
    indices = generator.integers(0, vector.size, size=(int(resamples), vector.size))
    medians = np.median(vector[indices], axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    return (
        float(np.quantile(medians, alpha, method="linear")),
        float(np.quantile(medians, 1.0 - alpha, method="linear")),
    )


PROTOCOL_METRICS = (
    "buffer_digest_ms",
    "durable_stage_once_ms",
    "verification_and_token_ms",
    "stage_and_issue_total_ms",
    "durable_commit_ms",
    "delivery_integrity_ms",
    "control_to_commit_ms",
    "complete_to_delivery_ms",
)


def summarize_protocol_timings(
    raw: pd.DataFrame,
    *,
    bootstrap_resamples: int,
) -> pd.DataFrame:
    group_columns = ["size_class", "dataset", "asset_id"]
    rows: list[dict[str, Any]] = []
    for keys, group in raw.groupby(group_columns, sort=False):
        row: dict[str, Any] = dict(zip(group_columns, keys, strict=True))
        row.update(
            {
                "timing_repetitions": int(len(group)),
                "answer_rows": int(group["answer_rows"].iloc[0]),
                "payload_bytes": int(group["payload_bytes"].iloc[0]),
                "stage_calls_min": int(group["stage_calls"].min()),
                "stage_calls_max": int(group["stage_calls"].max()),
                "release_rate": float(group["released"].mean()),
                "durable_rate": float(group["durable"].mean()),
            }
        )
        for metric in PROTOCOL_METRICS:
            values = group[metric].to_numpy(dtype=np.float64)
            low, high = _bootstrap_median_ci(
                values,
                resamples=bootstrap_resamples,
                seed=_stable_seed("protocol", *keys, metric),
            )
            prefix = metric.removesuffix("_ms")
            row[f"{prefix}_p50_ms"] = float(np.median(values))
            row[f"{prefix}_p95_observed_ms"] = float(
                np.quantile(values, 0.95, method="higher")
            )
            row[f"{prefix}_median_ci95_low_ms"] = low
            row[f"{prefix}_median_ci95_high_ms"] = high
        stage_seconds = max(float(np.median(group["durable_stage_once_ms"])) / 1000.0, 1e-12)
        delivery_seconds = max(
            float(np.median(group["delivery_integrity_ms"])) / 1000.0, 1e-12
        )
        payload_mib = int(group["payload_bytes"].iloc[0]) / (1024.0**2)
        row["stage_throughput_mib_s"] = payload_mib / stage_seconds
        row["delivery_verify_throughput_mib_s"] = payload_mib / delivery_seconds
        rows.append(row)
    return pd.DataFrame(rows)


def run_single_stage_protocol_benchmark(
    bundles: Mapping[str, Any],
    profiles: Mapping[str, Mapping[str, Any]],
    *,
    work_dir: Path,
    tolerance: float,
    repetitions: int,
    warmups: int,
    bootstrap_resamples: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Benchmark one-stage v1.1 issue/commit/delivery over actual answers.

    Store and schema construction occur outside each timed interval.  Every
    repetition uses a fresh SQLite ledger, preventing file-growth history from
    becoming an answer-size confounder.  Warmups are executed but never emitted.
    """

    if int(repetitions) < 1 or int(warmups) < 0:
        raise ValueError("invalid protocol timing repetition counts")
    work_dir.mkdir(parents=True, exist_ok=True)
    profile_names = [str(spec["size_class"]) for spec in ANSWER_PROFILE_SPECS]
    if set(profile_names) != set(profiles):
        raise ValueError("protocol benchmark did not receive the four profile answers")

    rows: list[dict[str, Any]] = []

    def one(profile_name: str, repetition: int, measured: bool, execution_order: int) -> None:
        profile = profiles[profile_name]
        dataset = str(profile["dataset"])
        bundle = bundles[dataset]
        asset_id = str(profile["asset_id"])
        asset_index = bundle.snapshot.asset_ids.index(asset_id)
        quoted_price = float(bundle.prices[asset_index])
        phase = "measured" if measured else "warmup"
        ledger_path = work_dir / f"{phase}-{repetition:04d}-{profile_name}.sqlite"
        for suffix in ["", "-wal", "-shm"]:
            Path(str(ledger_path) + suffix).unlink(missing_ok=True)
        ledger = InstrumentedDurableReleaseLedger(ledger_path)
        store = RequestBoundSnapshotStore(
            bundle.snapshot,
            ledger=ledger,
            asset_query_digests=bundle.query_digests,
            secret=secrets.token_bytes(32),
            verifier_tolerance=tolerance,
        )
        request = RequestContext.from_values(
            request_id=f"hardening-{phase}-{repetition:04d}-{profile_name}",
            buyer_id="hardening-buyer",
            session_id=f"hardening-{phase}-session",
            asset_id=asset_id,
            sql=str(profile["sql"]),
            quoted_price=quoted_price,
        )

        started = time.perf_counter_ns()
        buffer = BoundAnswerBuffer(request=request, payload=profile["answer"])
        buffer_ms = (time.perf_counter_ns() - started) / 1e6

        ledger.reset_stage_metrics()
        started = time.perf_counter_ns()
        token = store.issue_token(bundle.snapshot, bundle.prices, request, buffer)
        issue_total_ms = (time.perf_counter_ns() - started) / 1e6
        stage_calls = ledger.stage_calls
        stage_ms = ledger.stage_elapsed_ns / 1e6
        token_ms = issue_total_ms - stage_ms
        if stage_calls != 1 or token_ms < 0.0:
            raise RuntimeError(
                f"single-stage issue contract failed: calls={stage_calls}, residual_ms={token_ms}"
            )

        started = time.perf_counter_ns()
        commit = store.compare_and_commit(token, bundle.prices, request, buffer)
        commit_ms = (time.perf_counter_ns() - started) / 1e6

        started = time.perf_counter_ns()
        delivered = ledger.released_payload(token_id=token.token_id, request=request)
        delivery_ms = (time.perf_counter_ns() - started) / 1e6
        delivery_matches = delivered == buffer.payload_bytes
        if not commit.released or not commit.durable or not delivery_matches:
            raise RuntimeError("single-stage protocol benchmark failed release integrity")

        if measured:
            rows.append(
                {
                    "size_class": profile_name,
                    "dataset": dataset,
                    "asset_id": asset_id,
                    "repetition": int(repetition),
                    "execution_order": int(execution_order),
                    "answer_rows": int(profile["answer_rows"]),
                    "payload_bytes": int(len(buffer.payload_bytes)),
                    "stage_calls": int(stage_calls),
                    "duplicate_stage_detected": bool(stage_calls != 1),
                    "buffer_digest_ms": float(buffer_ms),
                    "durable_stage_once_ms": float(stage_ms),
                    "verification_and_token_ms": float(token_ms),
                    "stage_and_issue_total_ms": float(issue_total_ms),
                    "durable_commit_ms": float(commit_ms),
                    "delivery_integrity_ms": float(delivery_ms),
                    "control_to_commit_ms": float(buffer_ms + issue_total_ms + commit_ms),
                    "complete_to_delivery_ms": float(
                        buffer_ms + issue_total_ms + commit_ms + delivery_ms
                    ),
                    "released": bool(commit.released),
                    "durable": bool(commit.durable),
                    "delivery_matches_buffer": bool(delivery_matches),
                    "price_digest": token.price_digest,
                    "constraint_digest": token.constraint_digest,
                    "answer_digest": token.answer_digest,
                }
            )
        ledger.close_thread_connection()
        for suffix in ["", "-wal", "-shm"]:
            Path(str(ledger_path) + suffix).unlink(missing_ok=True)

    for warmup in range(int(warmups)):
        for order, name in enumerate(_balanced_profile_order(profile_names, warmup)):
            one(name, -warmup - 1, False, order)
    for repetition in range(int(repetitions)):
        for order, name in enumerate(_balanced_profile_order(profile_names, repetition)):
            one(name, repetition, True, order)

    raw = pd.DataFrame(rows)
    summary = summarize_protocol_timings(
        raw, bootstrap_resamples=int(bootstrap_resamples)
    )
    return raw, summary


@dataclass
class OptimizedTieredRepairResult:
    prices: np.ndarray
    verification: core.VerificationResult
    rejected: bool
    fallback_used: bool
    fallback_stage: str
    solver: str
    sweeps: int
    projected_rows: int
    runtime_ms: float
    fallback_ms: float
    l1_adjustment: float
    error: str


def run_optimized_tiered_fail_safe_repair(
    snapshot: core.ConstraintSnapshot,
    candidate: np.ndarray,
    *,
    tolerance: float,
    max_sweeps: int,
    active_row_budget: int,
    step: float,
    osqp_max_iter: int,
    violated_only_fallback_sweeps: int = 20,
) -> OptimizedTieredRepairResult:
    """Active repair -> violated-only cyclic -> OSQP -> reject."""

    original = np.asarray(candidate, dtype=np.float64)
    active = core.run_enforcement_method(
        snapshot,
        original,
        method="arbiter_active",
        tolerance=tolerance,
        max_sweeps=max_sweeps,
        active_row_budget=active_row_budget,
        step=step,
        allow_fallback=False,
        osqp_max_iter=osqp_max_iter,
    )
    if active.verification.passed:
        return OptimizedTieredRepairResult(
            prices=active.prices,
            verification=active.verification,
            rejected=False,
            fallback_used=False,
            fallback_stage="none",
            solver=active.solver,
            sweeps=active.sweeps,
            projected_rows=active.projected_rows,
            runtime_ms=active.runtime_ms,
            fallback_ms=0.0,
            l1_adjustment=float(np.abs(active.prices - original).sum()),
            error=active.error,
        )

    # The active kernel already iterates only rows violated at the beginning of
    # each sweep.  Expanding its budget to every current violation produces a
    # full-coverage fallback without the legacy loop over satisfied rows.
    violated_only = core.run_enforcement_method(
        snapshot,
        active.prices,
        method="arbiter_active",
        tolerance=tolerance,
        max_sweeps=violated_only_fallback_sweeps,
        active_row_budget=snapshot.constraints,
        step=step,
        allow_fallback=False,
        osqp_max_iter=osqp_max_iter,
    )
    total_runtime = active.runtime_ms + violated_only.runtime_ms
    total_sweeps = active.sweeps + violated_only.sweeps
    total_projected = active.projected_rows + violated_only.projected_rows
    if violated_only.verification.passed:
        return OptimizedTieredRepairResult(
            prices=violated_only.prices,
            verification=violated_only.verification,
            rejected=False,
            fallback_used=True,
            fallback_stage="violated_only_cyclic",
            solver="active_cyclic+violated_only_cyclic",
            sweeps=total_sweeps,
            projected_rows=total_projected,
            runtime_ms=total_runtime,
            fallback_ms=violated_only.runtime_ms,
            l1_adjustment=float(np.abs(violated_only.prices - original).sum()),
            error="; ".join(
                part for part in [active.error, violated_only.error] if part
            ),
        )

    qp = core.run_enforcement_method(
        snapshot,
        original,
        method="osqp_qp",
        tolerance=tolerance,
        max_sweeps=max_sweeps,
        active_row_budget=active_row_budget,
        step=step,
        allow_fallback=False,
        osqp_max_iter=osqp_max_iter,
    )
    final_verification = core.independent_verify(
        snapshot, qp.prices, tolerance=tolerance
    )
    rejected = not final_verification.passed
    return OptimizedTieredRepairResult(
        prices=qp.prices,
        verification=final_verification,
        rejected=rejected,
        fallback_used=True,
        fallback_stage="osqp" if not rejected else "rejected_after_osqp",
        solver="active_cyclic+violated_only_cyclic+osqp",
        sweeps=total_sweeps + qp.sweeps,
        projected_rows=total_projected + qp.projected_rows,
        runtime_ms=total_runtime + qp.runtime_ms,
        fallback_ms=violated_only.runtime_ms + qp.runtime_ms,
        l1_adjustment=float(np.abs(qp.prices - original).sum()),
        error="; ".join(
            part for part in [active.error, violated_only.error, qp.error] if part
        ),
    )


def _prior_fallback_conditions(
    cases: Sequence[dict[str, Any]],
    prior_budget_raw: pd.DataFrame,
) -> list[dict[str, Any]]:
    required = {
        "dataset",
        "relation_family",
        "case_id",
        "active_row_budget",
        "variant",
        "fallback_used",
        "fallback_stage",
        "oracle_verified",
    }
    if not required.issubset(prior_budget_raw.columns):
        raise ValueError("Notebook 05 budget evidence schema changed")
    fallback = prior_budget_raw[
        prior_budget_raw["variant"].astype(str).eq("fail_safe")
        & _boolean_series(
            prior_budget_raw["fallback_used"], name="Notebook 05 fallback flags"
        )
    ].copy()
    if fallback.empty:
        raise RuntimeError("Notebook 05 contains no exercised fallback condition")
    if not _boolean_series(
        fallback["oracle_verified"], name="Notebook 05 fallback verification"
    ).all():
        raise RuntimeError("Notebook 05 contains an unverified fallback release")
    if set(fallback["fallback_stage"].astype(str)) != {"full_cyclic"}:
        raise RuntimeError("Notebook 05 fallback stage is not the expected full cyclic path")
    key_columns = [
        "dataset",
        "relation_family",
        "case_id",
        "active_row_budget",
    ]
    unique = fallback[key_columns].drop_duplicates().sort_values(key_columns)
    case_map = {
        (str(case["dataset"]), str(case["relation_family"]), str(case["case_id"])): case
        for case in cases
    }
    conditions: list[dict[str, Any]] = []
    for row in unique.itertuples(index=False):
        key = (str(row.dataset), str(row.relation_family), str(row.case_id))
        if key not in case_map:
            raise RuntimeError(f"could not reconstruct Notebook 05 fallback case: {key}")
        conditions.append(
            {
                "dataset": key[0],
                "relation_family": key[1],
                "case_id": key[2],
                "active_row_budget": int(row.active_row_budget),
                "case": case_map[key],
            }
        )
    return conditions


FALLBACK_METHODS = (
    "legacy_full_scan_cyclic",
    "optimized_violated_only_cyclic",
)


def summarize_fallback_timings(
    raw: pd.DataFrame,
    *,
    bootstrap_resamples: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_columns = [
        "dataset",
        "relation_family",
        "case_id",
        "active_row_budget",
        "method",
    ]
    summary_rows: list[dict[str, Any]] = []
    for keys, group in raw.groupby(group_columns, sort=True):
        values = group["runtime_ms"].to_numpy(dtype=np.float64)
        low, high = _bootstrap_median_ci(
            values,
            resamples=bootstrap_resamples,
            seed=_stable_seed("fallback", *keys),
        )
        summary_rows.append(
            {
                **dict(zip(group_columns, keys, strict=True)),
                "timing_repetitions": int(len(group)),
                "runtime_p50_ms": float(np.median(values)),
                "runtime_p95_observed_ms": float(
                    np.quantile(values, 0.95, method="higher")
                ),
                "runtime_median_ci95_low_ms": low,
                "runtime_median_ci95_high_ms": high,
                "verified_rate": float(group["oracle_verified"].mean()),
                "unsafe_release_rate": float(group["unsafe_release"].mean()),
                "projected_rows_median": float(group["projected_rows"].median()),
                "sweeps_median": float(group["sweeps"].median()),
            }
        )
    summary = pd.DataFrame(summary_rows)

    key_columns = [
        "dataset",
        "relation_family",
        "case_id",
        "active_row_budget",
        "repetition",
    ]
    pivot = raw.pivot(index=key_columns, columns="method", values="runtime_ms").reset_index()
    if not set(FALLBACK_METHODS).issubset(pivot.columns):
        raise RuntimeError("paired fallback timing rows are incomplete")
    pivot["paired_speedup_legacy_over_optimized"] = (
        pivot["legacy_full_scan_cyclic"]
        / pivot["optimized_violated_only_cyclic"]
    )
    paired_rows: list[dict[str, Any]] = []
    condition_columns = key_columns[:-1]
    for keys, group in pivot.groupby(condition_columns, sort=True):
        values = group["paired_speedup_legacy_over_optimized"].to_numpy(float)
        low, high = _bootstrap_median_ci(
            values,
            resamples=bootstrap_resamples,
            seed=_stable_seed("fallback-speedup", *keys),
        )
        paired_rows.append(
            {
                **dict(zip(condition_columns, keys, strict=True)),
                "paired_repetitions": int(len(group)),
                "paired_speedup_p50": float(np.median(values)),
                "paired_speedup_p05_observed": float(
                    np.quantile(values, 0.05, method="lower")
                ),
                "paired_speedup_median_ci95_low": low,
                "paired_speedup_median_ci95_high": high,
            }
        )
    return summary, pd.DataFrame(paired_rows)


def run_fallback_kernel_comparison(
    cases: Sequence[dict[str, Any]],
    prior_budget_raw: pd.DataFrame,
    *,
    tolerance: float,
    max_sweeps: int,
    step: float,
    osqp_max_iter: int,
    repetitions: int,
    warmups: int,
    bootstrap_resamples: int,
    fallback_sweeps: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Paired legacy-vs-optimized timing from identical active intermediates."""

    if int(repetitions) < 1 or int(warmups) < 0:
        raise ValueError("invalid fallback timing repetition counts")
    conditions = _prior_fallback_conditions(cases, prior_budget_raw)
    rows: list[dict[str, Any]] = []

    def run_method(
        condition: Mapping[str, Any],
        active_prices: np.ndarray,
        method: str,
        repetition: int,
        execution_order: int,
        measured: bool,
    ) -> None:
        snapshot = condition["case"]["_snapshot"]
        if method == "legacy_full_scan_cyclic":
            core_method = "full_cyclic"
        elif method == "optimized_violated_only_cyclic":
            core_method = "arbiter_active"
        else:
            raise ValueError(f"unknown fallback method: {method}")
        started = time.perf_counter_ns()
        result = core.run_enforcement_method(
            snapshot,
            active_prices,
            method=core_method,
            tolerance=tolerance,
            max_sweeps=fallback_sweeps,
            active_row_budget=snapshot.constraints,
            step=step,
            allow_fallback=False,
            osqp_max_iter=osqp_max_iter,
        )
        runtime_ms = (time.perf_counter_ns() - started) / 1e6
        oracle = core.independent_verify(snapshot, result.prices, tolerance=tolerance)
        if not oracle.passed or result.rejected:
            raise RuntimeError(
                f"fallback kernel failed verification: {condition['case_id']}/{method}"
            )
        if measured:
            candidate = condition["case"]["_candidate"]
            rows.append(
                {
                    "dataset": condition["dataset"],
                    "relation_family": condition["relation_family"],
                    "case_id": condition["case_id"],
                    "active_row_budget": int(condition["active_row_budget"]),
                    "method": method,
                    "repetition": int(repetition),
                    "execution_order": int(execution_order),
                    "runtime_ms": float(runtime_ms),
                    "kernel_reported_ms": float(result.runtime_ms),
                    "sweeps": int(result.sweeps),
                    "projected_rows": int(result.projected_rows),
                    "oracle_verified": bool(oracle.passed),
                    "violations_after": int(oracle.violations),
                    "max_residual_after": float(oracle.max_residual),
                    "unsafe_release": bool(not result.rejected and not oracle.passed),
                    "l1_adjustment_from_candidate": float(
                        np.abs(result.prices - candidate).sum()
                    ),
                    "price_digest": core.price_vector_digest(
                        snapshot.asset_ids, result.prices
                    ),
                    "candidate_digest": condition["case"]["candidate_digest"],
                    "constraint_digest": snapshot.constraint_digest,
                }
            )

    for condition_index, condition in enumerate(conditions):
        case = condition["case"]
        snapshot = case["_snapshot"]
        active = core.run_enforcement_method(
            snapshot,
            case["_candidate"],
            method="arbiter_active",
            tolerance=tolerance,
            max_sweeps=max_sweeps,
            active_row_budget=int(condition["active_row_budget"]),
            step=step,
            allow_fallback=False,
            osqp_max_iter=osqp_max_iter,
        )
        if active.verification.passed:
            raise RuntimeError(
                f"Notebook 05 fallback condition no longer requires fallback: {case['case_id']}"
            )
        for warmup in range(int(warmups)):
            order = list(FALLBACK_METHODS)
            if (condition_index + warmup) % 2:
                order.reverse()
            for execution_order, method in enumerate(order):
                run_method(
                    condition,
                    active.prices,
                    method,
                    -warmup - 1,
                    execution_order,
                    False,
                )
        for repetition in range(int(repetitions)):
            order = list(FALLBACK_METHODS)
            if (condition_index + repetition) % 2:
                order.reverse()
            for execution_order, method in enumerate(order):
                run_method(
                    condition,
                    active.prices,
                    method,
                    repetition,
                    execution_order,
                    True,
                )

    raw = pd.DataFrame(rows)
    summary, paired = summarize_fallback_timings(
        raw, bootstrap_resamples=int(bootstrap_resamples)
    )
    return raw, summary, paired


def run_optimized_fail_safe_validation(
    cases: Sequence[dict[str, Any]],
    *,
    budgets: Sequence[int],
    tolerance: float,
    max_sweeps: int,
    step: float,
    osqp_max_iter: int,
) -> pd.DataFrame:
    """Run the optimized tier once over the complete dense case/budget grid."""

    rows: list[dict[str, Any]] = []
    for case in cases:
        snapshot = case["_snapshot"]
        candidate = case["_candidate"]
        precheck = core.independent_verify(snapshot, candidate, tolerance=tolerance)
        for budget in budgets:
            result = run_optimized_tiered_fail_safe_repair(
                snapshot,
                candidate,
                tolerance=tolerance,
                max_sweeps=max_sweeps,
                active_row_budget=int(budget),
                step=step,
                osqp_max_iter=osqp_max_iter,
            )
            oracle = core.independent_verify(
                snapshot, result.prices, tolerance=tolerance
            )
            rows.append(
                {
                    "dataset": case["dataset"],
                    "relation_family": case["relation_family"],
                    "case_id": case["case_id"],
                    "active_row_budget": int(budget),
                    "violations_before": int(precheck.violations),
                    "budget_binding": bool(precheck.violations > int(budget)),
                    "fallback_used": bool(result.fallback_used),
                    "fallback_stage": result.fallback_stage,
                    "solver": result.solver,
                    "rejected": bool(result.rejected),
                    "oracle_verified": bool(oracle.passed),
                    "violations_after": int(oracle.violations),
                    "max_residual_after": float(oracle.max_residual),
                    "unsafe_release": bool(not result.rejected and not oracle.passed),
                    "runtime_ms": float(result.runtime_ms),
                    "fallback_ms": float(result.fallback_ms),
                    "sweeps": int(result.sweeps),
                    "projected_rows": int(result.projected_rows),
                    "l1_adjustment": float(result.l1_adjustment),
                    "candidate_digest": case["candidate_digest"],
                    "constraint_digest": snapshot.constraint_digest,
                }
            )
    return pd.DataFrame(rows)


def hardening_quality_checks(
    *,
    mode: str,
    audited_input_audit: pd.DataFrame,
    closure_input_audit: pd.DataFrame,
    database_input_audit: pd.DataFrame,
    profile_catalog: pd.DataFrame,
    protocol_raw: pd.DataFrame,
    protocol_summary: pd.DataFrame,
    protocol_faults: pd.DataFrame,
    concurrency: pd.DataFrame,
    delivery: pd.DataFrame,
    fallback_raw: pd.DataFrame,
    fallback_summary: pd.DataFrame,
    fallback_paired: pd.DataFrame,
    optimized_grid: pd.DataFrame,
) -> pd.DataFrame:
    """Fail-closed evidence checks for Notebook 06."""

    if mode not in {"pilot", "full"}:
        raise ValueError("mode must be pilot or full")
    expected_protocol_repetitions = 30 if mode == "full" else 3
    expected_fallback_repetitions = 50 if mode == "full" else 3
    expected_concurrency_rows = 300 if mode == "full" else 18
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: Any) -> None:
        checks.append({"check": name, "passed": bool(passed)})

    add("all audited-v4 input checks pass", audited_input_audit["passed"].all())
    add("all Notebook 05 input checks pass", closure_input_audit["passed"].all())
    add(
        "observational source hashes and all ten database table counts pass",
        len(database_input_audit) == 12 and database_input_audit["passed"].all(),
    )
    add(
        "four preregistered actual DuckDB answer profiles are present",
        len(profile_catalog) == 4
        and set(profile_catalog["size_class"]) == {"scalar", "small", "medium", "large"}
        and set(profile_catalog["source"])
        == {"DuckDB execution over audited NYC/Adult/TPC-H database"},
    )
    band_pass = True
    spec_by_class = {str(item["size_class"]): item for item in ANSWER_PROFILE_SPECS}
    for row in profile_catalog.itertuples(index=False):
        spec = spec_by_class.get(str(row.size_class))
        band_pass = bool(
            band_pass
            and spec is not None
            and int(spec["min_rows"]) <= int(row.answer_rows) <= int(spec["max_rows"])
            and int(row.answer_rows) == int(spec["expected_rows"])
        )
    add("every answer profile satisfies its preregistered row band", band_pass)
    add(
        "protocol raw rows have unique profile/repetition keys",
        not protocol_raw.duplicated(["size_class", "repetition"]).any(),
    )
    protocol_profile_consistency = True
    for catalog_row in profile_catalog.itertuples(index=False):
        timed = protocol_raw[
            protocol_raw["size_class"].astype(str).eq(str(catalog_row.size_class))
        ]
        protocol_profile_consistency = bool(
            protocol_profile_consistency
            and not timed.empty
            and timed["answer_rows"].nunique() == 1
            and int(timed["answer_rows"].iloc[0]) == int(catalog_row.answer_rows)
            and timed["payload_bytes"].nunique() == 1
            and int(timed["payload_bytes"].iloc[0]) == int(catalog_row.payload_bytes)
        )
    add(
        "protocol raw answer rows and bytes match the executed profile catalog",
        protocol_profile_consistency,
    )
    add(
        "protocol timing uses the required repetitions per answer profile",
        len(protocol_raw) == 4 * expected_protocol_repetitions
        and protocol_raw.groupby("size_class")["repetition"].nunique().eq(
            expected_protocol_repetitions
        ).all(),
    )
    add(
        "every measured issue performs exactly one durable stage",
        protocol_raw["stage_calls"].eq(1).all()
        and not protocol_raw["duplicate_stage_detected"].any(),
    )
    add(
        "every measured protocol path releases durably with matching delivery",
        protocol_raw["released"].all()
        and protocol_raw["durable"].all()
        and protocol_raw["delivery_matches_buffer"].all(),
    )
    add(
        "protocol summary reconciles raw repetition counts",
        len(protocol_summary) == 4
        and protocol_summary["timing_repetitions"].eq(
            expected_protocol_repetitions
        ).all(),
    )
    add(
        "all 63 request-binding fault cases pass",
        len(protocol_faults) == 63 and protocol_faults["passed"].all(),
    )
    add(
        "all cross-service concurrency regressions pass",
        len(concurrency) == expected_concurrency_rows and concurrency["passed"].all(),
    )
    add(
        "all delivery-authorization regressions pass",
        not delivery.empty and delivery["passed"].all(),
    )
    fallback_conditions = fallback_raw[
        ["dataset", "relation_family", "case_id", "active_row_budget"]
    ].drop_duplicates()
    add(
        "the four Notebook 05 fallback conditions are reconstructed",
        len(fallback_conditions) == 4,
    )
    add(
        "fallback timing rows have complete paired repetitions",
        len(fallback_raw) == 4 * 2 * expected_fallback_repetitions
        and not fallback_raw.duplicated(
            [
                "dataset",
                "relation_family",
                "case_id",
                "active_row_budget",
                "method",
                "repetition",
            ]
        ).any(),
    )
    add(
        "both fallback kernels independently verify every timing result",
        fallback_raw["oracle_verified"].all()
        and not fallback_raw["unsafe_release"].any(),
    )
    add(
        "fallback summary reconciles raw repetitions",
        len(fallback_summary) == 8
        and fallback_summary["timing_repetitions"].eq(
            expected_fallback_repetitions
        ).all(),
    )
    add(
        "optimized fallback is faster in every paired condition",
        len(fallback_paired) == 4
        and fallback_paired["paired_speedup_p50"].gt(1.0).all()
        and fallback_paired["paired_speedup_median_ci95_low"].gt(1.0).all(),
    )
    add(
        "optimized fail-safe grid has no unverified release",
        len(optimized_grid) == 18 * 6
        and optimized_grid["oracle_verified"].all()
        and not optimized_grid["unsafe_release"].any(),
    )
    add(
        "optimized fail-safe retains OSQP-or-reject backstop semantics",
        set(optimized_grid["fallback_stage"]).issubset(
            {"none", "violated_only_cyclic", "osqp", "rejected_after_osqp"}
        ),
    )

    numeric_frames = {
        "profile_catalog": profile_catalog,
        "protocol_raw": protocol_raw,
        "protocol_summary": protocol_summary,
        "fallback_raw": fallback_raw,
        "fallback_summary": fallback_summary,
        "fallback_paired": fallback_paired,
        "optimized_grid": optimized_grid,
    }
    for name, frame in numeric_frames.items():
        numeric = frame.select_dtypes(include=[np.number])
        add(
            f"{name} contains no NaN or infinite numeric evidence",
            np.isfinite(numeric.to_numpy(dtype=float)).all(),
        )
    return pd.DataFrame(checks)


def build_claim_audit(
    *,
    mode: str,
    quality: pd.DataFrame,
) -> pd.DataFrame:
    passed = bool(quality["passed"].all())
    full = mode == "full"
    supported = "SUPPORTED" if passed and full else "NOT_SUPPORTED"
    return pd.DataFrame(
        [
            {
                "claim": "The measured v1.1 protocol path performs exactly one durable answer stage before request-bound commit.",
                "status": supported,
                "required_wording": "Limit to the four preregistered answer profiles and the recorded Colab environment.",
            },
            {
                "claim": "Buffering, durable staging, authorization, commit, and delivery-integrity costs are separately measured for scalar, small, medium, and large actual DuckDB answers.",
                "status": supported,
                "required_wording": "Report answer rows and serialized bytes; do not generalize beyond the four profiles.",
            },
            {
                "claim": "Violated-only cyclic fallback is faster than legacy full-row scanning on all four exercised Notebook 05 fallback conditions while preserving independent verification.",
                "status": supported,
                "required_wording": "Describe paired kernel-only measurements; do not call them end-to-end speedups.",
            },
            {
                "claim": "P95 latency generalizes to other machines or workload distributions.",
                "status": "NOT_SUPPORTED",
                "required_wording": "Treat observed P95 as descriptive for one controlled machine run; report n and the quantile rule.",
            },
            {
                "claim": "Network delivery, crash recovery, payment settlement, or distributed atomicity is established.",
                "status": "NOT_SUPPORTED",
                "required_wording": "The claim ends at a local SQLite authorization/outbox transaction and integrity-checked retrieval.",
            },
            {
                "claim": "General SQL or standard TPC-H benchmark-query coverage is established.",
                "status": "NOT_SUPPORTED",
                "required_wording": "Retain the audited supported-fragment boundary.",
            },
        ]
    )
