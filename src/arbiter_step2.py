"""ARBITER Step 2 campaign (Notebook 08), version 1.0.

Why this campaign exists
------------------------
The supervisor's review and the internal review of the PVLDB draft asked for
five additions that need new measurements:

* s21: GROUP BY with COUNT or SUM over one grouping column, with two new proof
  rules (grouped partition union and roll-up to the total) plus a group-slice
  rule, validated like R1 to R3 (independent oracle, executed checks in
  DuckDB) and evaluated under model proposals and per-row attacks.
* s22: an audit for relations that the declared rules miss. Rule R1 needs both
  parts of a two-part split to exist as assets; a target whose answer is a
  union of three or more existing intervals has no row. A shortest-path check
  over each predicate surface finds the cheapest chain of parts, flags undeclared
  savings, and can add the violated chain rows lazily. The stage also times the
  audit that checks a release against relations a deployment does not enforce.
* s23: the tolerance study under price floors, so that the smallest list price
  of a relation target varies over orders of magnitude.
* s24: the price change of each method measured against an exact projection
  reference (L1 and L2), with a non-ADMM interior-point solver (Clarabel).
* s25: scale-out beyond 4,096 assets per dataset with the same generator and
  catalog statistics, plus a combined marketplace of all three datasets.

Integrity rules
---------------
* The audited-v4 evidence package is read-only input and is verified by
  SHA-256 before any experiment starts (arbiter_revision.verify_audited_package).
* The audited modules, the closure modules, and the Notebook 07 module are
  imported unchanged. This module adds orchestration, the grouped-query
  extension, and measurement only.
* Measurement outcomes (rejection, fallback, solver failure) are recorded as
  findings. Only integrity failures (an unverified release, an incomplete grid,
  a compiler mismatch, a failed executed check, a changed input) fail the run.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sqlite3
import time
import traceback
from typing import Any, Callable, Mapping, Sequence
import zipfile

import numpy as np
import pandas as pd
from scipy import sparse

import arbiter_core as core
import arbiter_experiments as ex
import arbiter_revision as rev
import arbiter_workload as wl

try:
    import sqlglot
    from sqlglot import exp
except Exception:  # pragma: no cover - the notebook installs the pinned dependency.
    sqlglot = None
    exp = None


# ---------------------------------------------------------------------------
# Identity and configuration
# ---------------------------------------------------------------------------

STEP2_SUITE_VERSION = "2026-10-23-step2-v1.1"
SUITE_TAG = "v1_1"
PRIMARY_TOLERANCE = rev.PRIMARY_TOLERANCE
GROUPED_MODE = "grouped_aggregate"
AGGREGATE_MODES = ("additive_aggregate", GROUPED_MODE)
# Azcoitia, Iordanou, and Laoutaris (ICDE 2023, Section III) report a median
# price of US$2,176 for data products sold as a one-off purchase in commercial
# marketplaces (4,162 priced products from 443 providers). It is used only to
# express controlled price units in dollars, with one unit set to that median.
DOLLARS_PER_PRICE_UNIT = 2176.0
ASSET_COLUMNS = ["dataset", "asset_id", "sql", "family", "template_group", "expected_supported", "oracle_descriptor"]

STAGES = {
    "s21": "GROUP BY extension: grouped additive aggregates, new rules, validation, release",
    "s22": "multi-part partition audit, lazy enforcement, and audit cost",
    "s23": "tolerance study under price floors",
    "s24": "exact projection reference: L1 and L2 price change, non-ADMM solver",
    "s25": "scale-out beyond 4,096 assets and a combined marketplace",
}
STAGE_VERSIONS = {stage: "1.0" for stage in STAGES}
STAGE_ORDER = tuple(STAGES)
STAGE_HEADLINES = {
    "s21": "summary",
    "s22": "summary",
    "s23": "summary",
    "s24": "summary",
    "s25": "summary",
}

FAST_METHODS = ("reject_only", "arbiter_active")
CLARABEL_METHODS = ("clarabel_default", "clarabel_tight")

MODE_CONFIG: dict[str, dict[str, Any]] = {
    "pilot": {
        "seeds": [11],
        "repair_sweeps": 400,
        "sweeps_to_verify_cap": 3000,
        "bootstrap_resamples": 2000,
        "groupby_columns_per_table": 1,
        "groupby_sum_measures": 1,
        "groupby_numeric_columns": 1,
        "groupby_triples_per_column": 1,
        "groupby_category_pairs": 1,
        "groupby_max_new_assets": 200,
        "groupby_exec_checks_per_family": 2,
        "groupby_attack_budgets": [0.25],
        "groupby_attack_cases": 3,
        "multipart_widths": [5, 7],
        "multipart_targets_per_dataset": 6,
        "multipart_attack_budgets": [0.25],
        "multipart_max_rounds": 10,
        "audit_repetitions": 3,
        "audit_mode_declarations": ["D"],
        "audit_mode_proposals": 3,
        "price_floors": [1e-6, 1e-4, 1e-2],
        "floor_tolerances": [1e-7, 1e-9, 1e-12],
        "projection_methods": ["arbiter_active", "full_cyclic", "osqp_qp", "clarabel_default", "clarabel_tight"],
        "projection_repetitions": 1,
        "projection_warmups": 0,
        "projection_proposals": 3,
        "clarabel_default_tolerance": 1e-8,
        "clarabel_tight_tolerance": 1e-12,
        "clarabel_max_iter": 400,
        "scale_out_sizes": {"UCI Adult": [4096, 8192]},
        "scale_out_combined": [],
        "scale_out_cases_per_family": 1,
        "scale_out_combined_cases_per_family": 1,
        "scale_out_attack_budget": 0.25,
        "scale_out_methods": ["reject_only", "arbiter_active", "full_cyclic", "osqp_qp", "clarabel_default"],
        "scale_out_fast_repetitions": 2,
        "scale_out_fast_warmups": 1,
        "scale_out_slow_repetitions": 1,
        "scale_out_slow_warmups": 0,
        "runtime_go_limit_hours": 8.0,
    },
    "full": {
        "seeds": [11, 23, 37, 51, 73],
        "repair_sweeps": 400,
        "sweeps_to_verify_cap": 3000,
        "bootstrap_resamples": 10000,
        "groupby_columns_per_table": 3,
        "groupby_sum_measures": 1,
        "groupby_numeric_columns": 2,
        "groupby_triples_per_column": 2,
        "groupby_category_pairs": 3,
        "groupby_max_new_assets": 1500,
        "groupby_exec_checks_per_family": 12,
        "groupby_attack_budgets": [0.10, 0.25, 0.50],
        "groupby_attack_cases": 20,
        "multipart_widths": [5, 6, 8],
        "multipart_targets_per_dataset": 60,
        "multipart_attack_budgets": [0.10, 0.25, 0.50],
        "multipart_max_rounds": 10,
        "audit_repetitions": 20,
        "audit_mode_declarations": ["D", "PF"],
        "audit_mode_proposals": 15,
        "price_floors": [1e-6, 1e-4, 1e-2],
        "floor_tolerances": [1e-7, 1e-9, 1e-12],
        "projection_methods": ["arbiter_active", "full_cyclic", "osqp_qp", "clarabel_default", "clarabel_tight"],
        "projection_repetitions": 3,
        "projection_warmups": 1,
        "projection_proposals": 15,
        "clarabel_default_tolerance": 1e-8,
        "clarabel_tight_tolerance": 1e-12,
        "clarabel_max_iter": 400,
        "scale_out_sizes": {
            "NYC Taxi 2024-01": [4096, 8192],
            "TPC-H SF1": [4096, 8192, 12288],
            "UCI Adult": [4096, 8192, 12288],
        },
        "scale_out_combined": [4096, 8192],
        "scale_out_cases_per_family": 4,
        "scale_out_combined_cases_per_family": 2,
        "scale_out_attack_budget": 0.25,
        "scale_out_methods": ["reject_only", "arbiter_active", "full_cyclic", "osqp_qp", "clarabel_default"],
        "scale_out_fast_repetitions": 10,
        "scale_out_fast_warmups": 3,
        "scale_out_slow_repetitions": 3,
        "scale_out_slow_warmups": 1,
        "runtime_go_limit_hours": 8.0,
    },
}


def mode_config(mode: str, override: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if mode not in MODE_CONFIG:
        raise ValueError(f"RUN_MODE must be one of {sorted(MODE_CONFIG)}")
    config = json.loads(json.dumps(MODE_CONFIG[mode]))
    if override:
        unknown = sorted(set(override) - set(config))
        if unknown:
            raise ValueError(f"unknown configuration keys in override: {unknown}")
        config.update(json.loads(json.dumps(dict(override))))
    config["mode"] = mode
    config["primary_tolerance"] = PRIMARY_TOLERANCE
    config["step2_suite_version"] = STEP2_SUITE_VERSION
    config["dollars_per_price_unit"] = DOLLARS_PER_PRICE_UNIT
    return config


# ---------------------------------------------------------------------------
# Checkpoints keyed by this suite
# ---------------------------------------------------------------------------


class Step2Checkpointer(rev.Checkpointer):
    """Notebook 07's checkpointer, keyed by this suite's stages and version."""

    def key(self, part: str) -> dict[str, Any]:
        stage = part.split("/")[0]
        return {
            "part": part,
            "stage": stage,
            "stage_version": STAGE_VERSIONS[stage],
            "mode": self.mode,
            "input_digest": self.input_digest,
            "config_digest": self.config_digest,
            "step2_suite_version": STEP2_SUITE_VERSION,
        }

    def provenance(self) -> pd.DataFrame:
        rows = []
        for marker in sorted(self.parts_dir.rglob(self.MARKER)):
            try:
                meta = json.loads(marker.read_text(encoding="utf-8"))
            except Exception:
                continue
            key = meta.get("key", {})
            environment = meta.get("environment", {})
            rows.append(
                {
                    "part": key.get("part"),
                    "stage": key.get("stage"),
                    "valid_for_this_run": key == self.key(str(key.get("part", "s21/x")))
                    if key.get("stage") in STAGE_VERSIONS
                    else False,
                    "elapsed_s": meta.get("elapsed_s"),
                    "finished_utc": meta.get("finished_utc"),
                    "session_id": environment.get("session_id"),
                    "cpu_model": environment.get("cpu_model"),
                    "logical_cpu_count": environment.get("logical_cpu_count"),
                    "python": environment.get("python"),
                }
            )
        return pd.DataFrame(rows)


def _as_flag(series: pd.Series) -> pd.Series:
    """Booleans read back from checkpoint CSVs, whatever pandas made of them.

    When parts are concatenated, pandas 2 turns a boolean column that is empty
    in some parts into floats (True becomes 1.0), while pandas 3 keeps objects.
    This parser accepts both, and rejects anything else, including missing values.
    """

    def convert(value: Any) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, float, np.integer, np.floating)) and not (isinstance(value, float) and math.isnan(value)):
            if float(value) in (0.0, 1.0):
                return bool(value)
        text = str(value).strip().lower()
        if text in {"true", "1", "1.0"}:
            return True
        if text in {"false", "0", "0.0"}:
            return False
        raise ValueError(f"column {series.name!r} contains a non-boolean value: {value!r}")

    return series.map(convert).astype(bool)


class _OrderOnlyConnection:
    """Stands in for DuckDB so that default_dataset_specs reveals its column order."""

    class _Result:
        def __init__(self, sql: str) -> None:
            self.sql = sql

        def fetchone(self):
            return ([0.0, 1.0],) if "quantile_cont" in self.sql else (0,)

        def fetchall(self):
            return [("x", 1)]

    def execute(self, sql: str):
        return self._Result(sql)


def audited_table_specs(package_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """table_specs.json with the column order the audited generator used.

    The audited run saved the specifications with sorted keys, but generated
    its workload from the column order of arbiter_data.default_dataset_specs.
    The generator iterates columns in dictionary order, so the saved file alone
    yields a different workload; restoring the order reproduces the audited
    workload exactly (checked in build_step2_context).
    """

    import arbiter_data as data

    saved = json.loads((Path(package_dir) / "00_workload" / "table_specs.json").read_text(encoding="utf-8"))
    reference = data.default_dataset_specs(_OrderOnlyConnection(), tpch_scale_factor=1.0, quantile_count=33)
    restored: dict[str, list[dict[str, Any]]] = {}
    for dataset, tables in saved.items():
        by_table = {str(spec["table"]): spec for spec in reference[dataset]}
        restored[dataset] = []
        for spec in tables:
            order = by_table[str(spec["table"])]
            spec = dict(spec)
            for key in ("numeric_columns", "categorical_columns"):
                if set(order[key]) != set(spec[key]):
                    raise RuntimeError(f"{dataset}/{spec['table']}: {key} differ from the audited specification")
                spec[key] = {column: spec[key][column] for column in order[key]}
            if list(spec["output_columns"]) != list(order["output_columns"]) or list(spec["measure_columns"]) != list(order["measure_columns"]):
                raise RuntimeError(f"{dataset}/{spec['table']}: output or measure columns differ from the audited specification")
            restored[dataset].append(spec)
    return restored


def build_step2_context(
    *,
    mode: str,
    config: dict[str, Any],
    package_dir: Path,
    output_dir: Path,
    input_audit: pd.DataFrame,
    package_state_before: dict[str, str],
    session_id: str,
    lock: rev.RunLock | None,
    log_fn: Callable[[str], None] = print,
):
    """Build Notebook 07's context over audited-v4 and attach Step 2 state."""

    ctx = rev.build_context(
        mode=mode,
        config=config,
        package_dir=package_dir,
        output_dir=output_dir,
        input_audit=input_audit,
        package_state_before=package_state_before,
        session_id=session_id,
        lock=lock,
        log_fn=log_fn,
    )
    # Derived files are rebuilt from checkpoints in every session; removing them
    # first keeps files of an earlier session or version out of the evidence ZIP.
    import shutil

    for name in ("outputs", "errors", "code"):
        shutil.rmtree(Path(output_dir) / name, ignore_errors=True)
    for name in ("stage_quality_checks.csv", "input_audit.csv", "part_provenance.csv", "runtime_estimate.csv",
                 "headline_numbers.json", "step2_run_config.json", "environment.json", "README.md",
                 "database_provenance.csv", "checksums.csv", "checksums.sha256"):
        (Path(output_dir) / name).unlink(missing_ok=True)
    ctx.checkpointer = Step2Checkpointer(
        Path(output_dir) / "parts",
        mode=mode,
        input_digest=rev.AUDITED_MANIFEST_SHA256,
        config_digest=rev.canonical_digest(config),
        session_id=session_id,
    )
    all_assets = rev.read_csv(Path(package_dir) / "00_workload" / "sql_assets.csv")
    ctx.step2 = {
        "table_specs": audited_table_specs(package_dir),
        "audited_relations": rev.read_csv(Path(package_dir) / "00_workload" / "compiled_relations.csv"),
        "negative_assets": all_assets[~_as_flag(all_assets["expected_supported"].astype(str))].reset_index(drop=True),
        "connection": None,
        "data_paths": {},
        "local_test": False,
    }
    if sqlglot is None:
        raise RuntimeError("SQLGlot did not load; the GROUP BY compiler cannot run. Restart the runtime and Run all.")
    for dataset, state in ctx.datasets.items():
        regenerated = wl.generate_deterministic_workload(
            dataset=dataset, table_specs=ctx.step2["table_specs"][dataset], max_assets=len(state.asset_ids))
        audited = all_assets[all_assets["dataset"] == dataset]
        same = (regenerated["asset_id"].astype(str).tolist() == audited["asset_id"].astype(str).tolist()
                and regenerated["sql"].astype(str).tolist() == audited["sql"].astype(str).tolist())
        ctx.add_check("setup", f"{dataset}: restored table specifications regenerate the audited workload", same)
        if not same:
            raise RuntimeError(f"{dataset}: the restored table specifications do not regenerate the audited workload")
    return ctx


# ---------------------------------------------------------------------------
# Grouped additive aggregates: parser, renderer, rules
# ---------------------------------------------------------------------------


def _grouped_descriptor(tree) -> wl.QueryDescriptor:
    """Parse SELECT g, COUNT(*)|SUM(m) FROM t [WHERE p] GROUP BY g, fail closed."""

    forbidden_nodes = {"CTE", "Lateral", "Pivot", "Subquery", "TableSample", "Unnest", "Window", "With"}
    if any(type(node).__name__ in forbidden_nodes for node in tree.walk()):
        raise ValueError("nested, sampled, or windowed sources are outside the certified fragment")
    for argument in ("qualify", "windows", "laterals", "pivots", "sample", "with", "with_", "connect", "match"):
        if tree.args.get(argument):
            raise ValueError(f"{argument} is outside the certified fragment")
    if tree.args.get("joins"):
        raise ValueError("joins are outside the certified fragment")
    if tree.args.get("having") is not None:
        raise ValueError("HAVING is outside the certified fragment")
    if tree.args.get("limit") is not None or tree.args.get("offset") is not None:
        raise ValueError("LIMIT/OFFSET is outside the certified fragment")
    if tree.args.get("order") is not None:
        raise ValueError("ORDER BY is outside the certified fragment")
    if tree.args.get("distinct") is not None:
        raise ValueError("DISTINCT is outside the certified fragment")
    group = tree.args.get("group")
    for option in ("rollup", "cube", "grouping_sets", "totals", "all"):
        if group.args.get(option):
            raise ValueError("ROLLUP, CUBE, GROUPING SETS, and GROUP BY ALL are outside the certified fragment")
    keys = list(group.expressions)
    if len(keys) != 1 or not isinstance(keys[0], exp.Column):
        raise ValueError("exactly one grouping column is required")
    group_column = str(keys[0].name)
    tables = list(tree.find_all(exp.Table))
    if len(tables) != 1:
        raise ValueError("exactly one base table is required")
    from_clause = tree.args.get("from_") or tree.args.get("from")
    if from_clause is None or not isinstance(from_clause.this, exp.Table):
        raise ValueError("FROM must reference one direct base table")
    table_expression = tables[0]
    if table_expression.args.get("db") or table_expression.args.get("catalog"):
        raise ValueError("schema/catalog-qualified tables are outside the certified fragment")
    table = str(table_expression.name)
    allowed_qualifiers = {table}
    if table_expression.alias:
        allowed_qualifiers.add(str(table_expression.alias))
    for column in tree.find_all(exp.Column):
        qualifier = str(column.table or "")
        if qualifier and qualifier not in allowed_qualifiers:
            raise ValueError("column qualifier does not name the direct base table")
    predicate = wl._parse_predicate(tree)
    raw_expressions = list(tree.expressions)
    if len(raw_expressions) != 2:
        raise ValueError("a grouped query selects its grouping column and one additive aggregate")
    first, second = raw_expressions
    if isinstance(first, exp.Alias) or not isinstance(first, exp.Column) or str(first.name) != group_column:
        raise ValueError("the first SELECT item must be the grouping column, without an alias")
    aggregate_node = wl._strip_alias(second)
    if isinstance(aggregate_node, exp.Count):
        if not isinstance(aggregate_node.this, exp.Star):
            raise ValueError("only COUNT(*) is supported")
        aggregate, measure = "count", "*"
    elif isinstance(aggregate_node, exp.Sum):
        aggregate, measure = "sum", wl._column_name(aggregate_node.this)
    elif isinstance(aggregate_node, exp.Avg):
        raise ValueError("AVG is not additive without sufficient statistics")
    else:
        raise ValueError("the second SELECT item must be COUNT(*) or SUM(column)")
    return wl.QueryDescriptor(
        table=table,
        mode=GROUPED_MODE,
        output_columns=(group_column,),
        aggregate=aggregate,
        measure_column=measure,
        **predicate,
    )


def descriptor_from_sql_ext(sql: str) -> wl.QueryDescriptor:
    """The audited parser, extended to grouped additive aggregates."""

    if sqlglot is None or exp is None:
        raise RuntimeError("sqlglot is required for SQL compilation")
    tree = sqlglot.parse_one(sql, read="duckdb")
    if isinstance(tree, exp.Select) and tree.args.get("group") is not None:
        return _grouped_descriptor(tree)
    return wl.descriptor_from_sql(sql)


def descriptor_to_sql_ext(descriptor: wl.QueryDescriptor) -> str:
    if descriptor.mode != GROUPED_MODE:
        return wl.descriptor_to_sql(descriptor)
    twin = replace(descriptor, mode="additive_aggregate", output_columns=())
    ungrouped = wl.descriptor_to_sql(twin)
    if not ungrouped.startswith("SELECT "):
        raise AssertionError("unexpected rendering of an additive aggregate")
    group_column = wl._quote_identifier(descriptor.output_columns[0])
    return f"SELECT {group_column}, {ungrouped[len('SELECT '):]} GROUP BY {group_column}"


PARSER_SELF_TEST = [
    # Supported shapes.
    ('SELECT "VendorID", COUNT(*) AS value FROM "nyc_taxi" GROUP BY "VendorID"', True),
    ('SELECT "VendorID", COUNT(*) AS n FROM "nyc_taxi" GROUP BY "VendorID"', True),
    ('SELECT "payment_type", SUM("fare_amount") AS value FROM "nyc_taxi" WHERE "trip_distance" >= 0.3 AND "trip_distance" < 0.5 GROUP BY "payment_type"', True),
    ('SELECT "payment_type", COUNT(*) AS value FROM "nyc_taxi" WHERE "payment_type" IN (\'1\', \'2\') GROUP BY "payment_type"', True),
    ('SELECT t."VendorID", COUNT(*) FROM "nyc_taxi" AS t GROUP BY t."VendorID"', True),
    ('SELECT COUNT(*) AS value FROM "nyc_taxi" WHERE "payment_type" IN (\'1\')', True),
    # Fail-closed shapes.
    ('SELECT "VendorID", "payment_type", COUNT(*) FROM "nyc_taxi" GROUP BY "VendorID", "payment_type"', False),
    ('SELECT "VendorID", COUNT(*) FROM "nyc_taxi" GROUP BY "VendorID" HAVING COUNT(*) > 5', False),
    ('SELECT "VendorID", AVG("fare_amount") FROM "nyc_taxi" GROUP BY "VendorID"', False),
    ('SELECT "VendorID", COUNT(*) FROM "nyc_taxi" GROUP BY "VendorID" ORDER BY "VendorID"', False),
    ('SELECT "VendorID", COUNT(*) FROM "nyc_taxi" GROUP BY "VendorID" LIMIT 3', False),
    ('SELECT DISTINCT "VendorID", COUNT(*) FROM "nyc_taxi" GROUP BY "VendorID"', False),
    ('SELECT "VendorID", COUNT(*) FROM "nyc_taxi" GROUP BY ROLLUP ("VendorID")', False),
    ('SELECT "VendorID", COUNT(*) FROM "nyc_taxi" GROUP BY CUBE ("VendorID")', False),
    ('SELECT "VendorID", COUNT(*) FROM "nyc_taxi" GROUP BY GROUPING SETS (("VendorID"))', False),
    ('SELECT "VendorID", COUNT(*) FROM "nyc_taxi" GROUP BY ALL', False),
    ('SELECT "VendorID", COUNT(*) FROM "nyc_taxi" GROUP BY 1', False),
    ('SELECT "payment_type", COUNT(*) FROM "nyc_taxi" GROUP BY "VendorID"', False),
    ('SELECT "VendorID" AS v, COUNT(*) FROM "nyc_taxi" GROUP BY "VendorID"', False),
    ('SELECT COUNT(*), "VendorID" FROM "nyc_taxi" GROUP BY "VendorID"', False),
    ('SELECT "VendorID", COUNT(*), SUM("fare_amount") FROM "nyc_taxi" GROUP BY "VendorID"', False),
    ('SELECT "VendorID", COUNT(DISTINCT "payment_type") FROM "nyc_taxi" GROUP BY "VendorID"', False),
    ('SELECT "VendorID", SUM(DISTINCT "fare_amount") FROM "nyc_taxi" GROUP BY "VendorID"', False),
    ('SELECT "VendorID", SUM("fare_amount" * 2) FROM "nyc_taxi" GROUP BY "VendorID"', False),
    ('SELECT "VendorID", SUM("fare_amount") FILTER (WHERE "fare_amount" > 0) FROM "nyc_taxi" GROUP BY "VendorID"', False),
    ('SELECT "VendorID", SUM("fare_amount") FROM "nyc_taxi" WHERE "fare_amount" > 3 GROUP BY "VendorID"', False),
    ('SELECT "VendorID", COUNT(*) FROM main."nyc_taxi" GROUP BY "VendorID"', False),
    ('SELECT "VendorID", SUM("fare_amount") FROM (SELECT * FROM "nyc_taxi") AS t GROUP BY "VendorID"', False),
    ('WITH q AS (SELECT * FROM "nyc_taxi") SELECT "VendorID", COUNT(*) FROM q GROUP BY "VendorID"', False),
    ('SELECT t."VendorID", COUNT(*) FROM "nyc_taxi" AS t JOIN "nyc_taxi" AS u ON t."VendorID" = u."VendorID" GROUP BY t."VendorID"', False),
    ('SELECT "VendorID", COUNT(*) FROM "nyc_taxi" GROUP BY "VendorID" UNION ALL SELECT "VendorID", COUNT(*) FROM "nyc_taxi" GROUP BY "VendorID"', False),
]


def parser_self_test() -> pd.DataFrame:
    """Fail-closed behavior of the extended parser on fixed strings."""

    rows = []
    for sql, expected in PARSER_SELF_TEST:
        try:
            descriptor = descriptor_from_sql_ext(sql)
            supported, reason = True, descriptor.to_json()
        except Exception as error:  # noqa: BLE001 - the reason is the result
            supported, reason = False, f"{type(error).__name__}: {error}"
        rows.append({"sql": sql, "expected_supported": expected, "parsed_supported": supported,
                     "passed": supported == expected, "reason": reason})
    frame = pd.DataFrame(rows)
    # The renderer must round-trip through the parser.
    probe = wl.QueryDescriptor(table="nyc_taxi", mode=GROUPED_MODE, output_columns=("VendorID",),
                               aggregate="sum", measure_column="fare_amount", predicate_kind="numeric_interval",
                               predicate_column="trip_distance", lower=0.3, upper=0.5, categories=())
    try:
        round_trip = descriptor_from_sql_ext(descriptor_to_sql_ext(probe)) == probe
        reason = descriptor_to_sql_ext(probe)
    except Exception as error:  # noqa: BLE001
        round_trip, reason = False, f"{type(error).__name__}: {error}"
    frame.loc[len(frame)] = {"sql": "renderer round trip", "expected_supported": True,
                             "parsed_supported": round_trip, "passed": round_trip, "reason": reason}
    return frame


# Relation encodings: the audited snapshot builder knows three relation types.
# A roll-up, a group slice, and a group filter are all "derived <= source"
# rows, so they are encoded with the derivation types and named by their rule.
RULE_ENCODING = {
    "grouped_partition_union": "decomposition",
    "group_rollup": "projection_derivation",
    "group_slice": "filter_derivation",
    "group_filter": "filter_derivation",
}
NEW_FAMILIES = ("grouped_partition_union", "partition_union_new_assets", "group_rollup", "group_slice", "group_filter")
DECOMPOSITION_FAMILIES = ("grouped_partition_union", "partition_union_new_assets")
RELATION_COLUMNS = ["dataset", "relation_id", "relation_type", "target_id", "component_ids",
                    "derived_id", "source_id", "sound", "proof_rule"]


def _derivation_row(dataset: str, prefix: str, rule: str, derived_id: str, source_id: str) -> dict[str, Any]:
    return {"dataset": dataset, "relation_id": wl._stable_id(prefix, dataset, derived_id, source_id),
            "relation_type": RULE_ENCODING[rule.replace("oracle_", "")], "target_id": "", "component_ids": "[]",
            "derived_id": derived_id, "source_id": source_id, "sound": True, "proof_rule": rule}


def grouped_derivation_relations(assets: pd.DataFrame, descriptors: Mapping[str, wl.QueryDescriptor]) -> pd.DataFrame:
    """Compiler rules for grouped aggregates, by index lookup (not pairwise).

    Let s be a grouped aggregate over table T with grouping column g, aggregate
    a, measure m, and predicate P.

    * Roll-up: the ungrouped aggregate (T, a, m, P) is the sum of s over all
      groups, so its price may not exceed the price of s.
    * Slice: when P is absent or is a category set on g, the ungrouped
      aggregate (T, a, m, g IN S) with S a strict subset of P's set (any
      nonempty S when P is absent) is the sum of s over the groups in S.
    * Group filter: under the same condition on P, the grouped aggregate
      (T, g, a, m, g IN S) is the subset of the rows of s whose group is in S.
    """

    rows = []
    for dataset, frame in assets.groupby("dataset", sort=False):
        records = [(str(a), descriptors[str(a)]) for a in frame["asset_id"].astype(str) if str(a) in descriptors]
        ungrouped_same: dict[tuple, list[str]] = defaultdict(list)
        on_column: dict[tuple, list[tuple[str, wl.QueryDescriptor]]] = defaultdict(list)
        for asset_id, d in records:
            if d.mode == "additive_aggregate":
                ungrouped_same[(d.table, d.aggregate, d.measure_column, d.predicate_key())].append(asset_id)
                if d.predicate_kind == "category_set":
                    on_column[("ungrouped", d.table, d.aggregate, d.measure_column, d.predicate_column)].append((asset_id, d))
            elif d.mode == GROUPED_MODE and d.predicate_kind == "category_set" and d.predicate_column == d.output_columns[0]:
                on_column[("grouped", d.table, d.aggregate, d.measure_column, d.predicate_column)].append((asset_id, d))
        for source_id, s in records:
            if s.mode != GROUPED_MODE:
                continue
            for derived_id in ungrouped_same.get((s.table, s.aggregate, s.measure_column, s.predicate_key()), []):
                rows.append(_derivation_row(dataset, "rollup", "group_rollup", derived_id, source_id))
            group_column = s.output_columns[0]
            if s.predicate_kind == "all":
                outer: set[str] | None = None
            elif s.predicate_kind == "category_set" and s.predicate_column == group_column:
                outer = set(s.categories)
            else:
                continue
            for kind, prefix, rule in (("ungrouped", "slice", "group_slice"), ("grouped", "groupfilter", "group_filter")):
                for derived_id, d in on_column.get((kind, s.table, s.aggregate, s.measure_column, group_column), []):
                    if derived_id != source_id and (outer is None or set(d.categories) < outer):
                        rows.append(_derivation_row(dataset, prefix, rule, derived_id, source_id))
    if not rows:
        return pd.DataFrame(columns=RELATION_COLUMNS)
    return pd.DataFrame(rows, columns=RELATION_COLUMNS).drop_duplicates("relation_id").reset_index(drop=True)


def infer_relations_ext(assets: pd.DataFrame, descriptors: Mapping[str, wl.QueryDescriptor]) -> pd.DataFrame:
    """Audited inference (R1 to R3, which also finds grouped partitions) plus the grouped rules."""

    base = wl.infer_relations_from_descriptors(assets, descriptors).copy()
    if len(base):
        grouped_targets = base["target_id"].astype(str).map(
            lambda target: target in descriptors and descriptors[target].mode == GROUPED_MODE
        )
        base.loc[grouped_targets & (base["relation_type"] == "decomposition"), "proof_rule"] = "grouped_partition_union"
    extra = grouped_derivation_relations(assets, descriptors)
    combined = pd.concat([base, extra], ignore_index=True, sort=False)
    return combined.drop_duplicates("relation_id").sort_values(["dataset", "relation_type", "relation_id"]).reset_index(drop=True)


def oracle_grouped_derivations(assets: pd.DataFrame) -> pd.DataFrame:
    """Independent pairwise oracle for the grouped rules, from construction descriptors.

    It compares every grouped asset with every aggregate asset of the same
    dataset and decides membership from answer containment, without the index
    structure that the compiler uses.
    """

    rows = []
    for dataset, frame in assets.groupby("dataset", sort=False):
        records = [(str(r.asset_id), wl.QueryDescriptor.from_json(str(r.oracle_descriptor)))
                   for r in frame.itertuples(index=False) if str(r.oracle_descriptor)]
        grouped = [(a, d) for a, d in records if d.mode == GROUPED_MODE]
        aggregates = [(a, d) for a, d in records if d.mode in AGGREGATE_MODES]
        for source_id, s in grouped:
            g = s.output_columns[0]
            for derived_id, d in aggregates:
                if derived_id == source_id or (s.table, s.aggregate, s.measure_column) != (d.table, d.aggregate, d.measure_column):
                    continue
                same_predicate = (
                    s.predicate_kind == d.predicate_kind and s.predicate_column == d.predicate_column
                    and s.lower == d.lower and s.upper == d.upper and tuple(s.categories) == tuple(d.categories)
                )
                if d.mode == "additive_aggregate" and same_predicate:
                    rows.append(_derivation_row(dataset, "oracle_rollup", "oracle_group_rollup", derived_id, source_id))
                    continue
                if d.mode == GROUPED_MODE and d.output_columns != s.output_columns:
                    continue
                if d.predicate_kind != "category_set" or d.predicate_column != g:
                    continue
                if s.predicate_kind == "all":
                    contained = True
                elif s.predicate_kind == "category_set" and s.predicate_column == g:
                    contained = set(d.categories) < set(s.categories)
                else:
                    contained = False
                if contained:
                    if d.mode == "additive_aggregate":
                        rows.append(_derivation_row(dataset, "oracle_slice", "oracle_group_slice", derived_id, source_id))
                    else:
                        rows.append(_derivation_row(dataset, "oracle_groupfilter", "oracle_group_filter", derived_id, source_id))
    if not rows:
        return pd.DataFrame(columns=RELATION_COLUMNS)
    return pd.DataFrame(rows, columns=RELATION_COLUMNS)


def relation_key(record: Mapping[str, Any]) -> tuple:
    return wl._relation_key(record)


def relation_assets(record: Mapping[str, Any]) -> set[str]:
    if str(record["relation_type"]) == "decomposition":
        return {str(record["target_id"]), *[str(item) for item in json.loads(str(record["component_ids"]))]}
    return {str(record["derived_id"]), str(record["source_id"])}


# ---------------------------------------------------------------------------
# Grouped workload generation (deterministic, from audited catalog statistics)
# ---------------------------------------------------------------------------


def _evenly_spaced(count: int, available: int) -> list[int]:
    if available <= 0 or count <= 0:
        return []
    if count >= available:
        return list(range(available))
    return sorted({int(round(i * (available - 1) / max(count - 1, 1))) for i in range(count)})


def generate_grouped_assets(dataset: str, table_specs: Sequence[Mapping[str, Any]], existing_ids: set[str],
                            cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Grouped aggregates and their ungrouped twins, in whole groups up to a cap.

    Every group closes under the new rules: G1 pairs the grouped answer over
    all rows with its ungrouped total; G2 adds a grouped interval partition
    (target and two parts) with the three ungrouped twins; G3 adds a grouped
    category partition on the grouping column with its three ungrouped twins.
    Groups are taken from the tables in turn until the cap is reached, and
    assets that already exist in the audited workload are not duplicated.
    """

    table_groups: dict[str, list[list[tuple[wl.QueryDescriptor, str, str]]]] = defaultdict(list)

    def record(table, *, mode, aggregate, measure, predicate_kind="all", predicate_column="",
               lower=None, upper=None, categories=(), group_column=None) -> wl.QueryDescriptor:
        return wl.QueryDescriptor(
            table=table, mode=mode,
            output_columns=(group_column,) if group_column else (),
            aggregate=aggregate, measure_column=measure, predicate_kind=predicate_kind,
            predicate_column=predicate_column, lower=lower, upper=upper,
            categories=tuple(sorted(str(item) for item in categories)),
        )

    for spec in table_specs:
        table = str(spec["table"])
        categorical = {str(c): [str(v) for v in values] for c, values in dict(spec.get("categorical_columns", {})).items()}
        numeric = {str(c): sorted(set(float(v) for v in values if np.isfinite(v)))
                   for c, values in dict(spec.get("numeric_columns", {})).items()}
        measures = [str(m) for m in spec.get("measure_columns", [])]
        aggregates = [("count", "*")] + [("sum", m) for m in measures[: int(cfg["groupby_sum_measures"])]]
        group_columns = list(categorical)[: int(cfg["groupby_columns_per_table"])]
        numeric_columns = [c for c in numeric if len(numeric[c]) >= 3][: int(cfg["groupby_numeric_columns"])]
        for g in group_columns:
            for aggregate, measure in aggregates:
                tag = f"{table}::grouped::{g}::{aggregate}:{measure}"
                table_groups[table].append([
                    (record(table, mode=GROUPED_MODE, aggregate=aggregate, measure=measure, group_column=g),
                     "grouped_all", f"{tag}::all"),
                    (record(table, mode="additive_aggregate", aggregate=aggregate, measure=measure),
                     "ungrouped_all", f"{table}::{aggregate}:{measure}::all"),
                ])
                for c in numeric_columns:
                    boundaries = numeric[c]
                    for start in _evenly_spaced(int(cfg["groupby_triples_per_column"]), len(boundaries) - 2):
                        lower, middle, upper = boundaries[start:start + 3]
                        if not lower < middle < upper:
                            continue
                        intervals = [(lower, upper), (lower, middle), (middle, upper)]
                        group = []
                        for position, (a, b) in enumerate(intervals):
                            group.append((record(table, mode=GROUPED_MODE, aggregate=aggregate, measure=measure,
                                                 predicate_kind="numeric_interval", predicate_column=c,
                                                 lower=a, upper=b, group_column=g),
                                          "grouped_partition_target" if position == 0 else "grouped_partition_component",
                                          f"{tag}::{c}"))
                        for a, b in intervals:
                            group.append((record(table, mode="additive_aggregate", aggregate=aggregate, measure=measure,
                                                 predicate_kind="numeric_interval", predicate_column=c, lower=a, upper=b),
                                          "ungrouped_twin", f"{table}::{aggregate}:{measure}::{c}"))
                        table_groups[table].append(group)
                pairs = [(left, right) for i, left in enumerate(categorical[g][:12]) for right in categorical[g][i + 1:12]]
                for pair_index in _evenly_spaced(int(cfg["groupby_category_pairs"]), len(pairs)):
                    left, right = pairs[pair_index]
                    sets = [(left, right), (left,), (right,)]
                    group = []
                    for position, values in enumerate(sets):
                        group.append((record(table, mode=GROUPED_MODE, aggregate=aggregate, measure=measure,
                                             predicate_kind="category_set", predicate_column=g, categories=values,
                                             group_column=g),
                                      "grouped_category_target" if position == 0 else "grouped_category_component",
                                      f"{tag}::{g}"))
                    for position, values in enumerate(sets):
                        group.append((record(table, mode="additive_aggregate", aggregate=aggregate, measure=measure,
                                             predicate_kind="category_set", predicate_column=g, categories=values),
                                      "ungrouped_category_target" if position == 0 else "ungrouped_category_component",
                                      f"{table}::{aggregate}:{measure}::{g}"))
                    table_groups[table].append(group)

    cap = int(cfg["groupby_max_new_assets"])
    rows: list[dict[str, Any]] = []
    seen = set(existing_ids)
    queues = [list(table_groups[str(spec["table"])]) for spec in table_specs]
    ordered_groups = []
    while any(queues):
        for queue in queues:
            if queue:
                ordered_groups.append(queue.pop(0))
    for group in ordered_groups:
        candidates = []
        for descriptor, family, template_group in group:
            sql = " ".join(descriptor_to_sql_ext(descriptor).split())
            asset_id = wl._stable_id("asset", dataset, sql)
            if asset_id in seen or any(c["asset_id"] == asset_id for c in candidates):
                continue
            candidates.append({"dataset": dataset, "asset_id": asset_id, "sql": sql, "family": family,
                               "template_group": template_group, "expected_supported": True,
                               "oracle_descriptor": descriptor.to_json()})
        if len(rows) + len(candidates) > cap:
            break
        rows.extend(candidates)
        seen.update(c["asset_id"] for c in candidates)
    return pd.DataFrame(rows, columns=["dataset", "asset_id", "sql", "family", "template_group",
                                       "expected_supported", "oracle_descriptor"])


# ---------------------------------------------------------------------------
# Controlled prices with the audited formula and fixed scales
# ---------------------------------------------------------------------------


def base_price_scales(features: pd.DataFrame) -> tuple[float, float]:
    interval = features["interval_width"].to_numpy(float)
    positive = interval[interval > 0]
    interval_scale = float(positive.max()) if positive.size else 1.0
    category_scale = max(float(features["category_count"].to_numpy(float).max(initial=1.0)), 1.0)
    return interval_scale, category_scale


def controlled_policy_fixed_scale(features: pd.DataFrame, interval_scale: float, category_scale: float) -> np.ndarray:
    """The audited linear price function with given scales; grouped aggregates get the aggregate multiplier."""

    predicate_kind = features["predicate_kind"].to_numpy(str)
    interval = features["interval_width"].to_numpy(float)
    category_count = features["category_count"].to_numpy(float)
    extent = np.where(
        predicate_kind == "numeric_interval",
        interval / max(interval_scale, 1e-12),
        np.where(predicate_kind == "category_set", category_count / category_scale, 1.0),
    )
    extent = np.maximum(extent, 1e-9)
    output_multiplier = 1.0 + 0.12 * features["output_count"].to_numpy(float)
    aggregate_multiplier = np.where(np.isin(features["mode"].to_numpy(str), AGGREGATE_MODES), 1.25, 1.0)
    return (extent * output_multiplier * aggregate_multiplier).astype(np.float64)


# ---------------------------------------------------------------------------
# Databases for executed checks
# ---------------------------------------------------------------------------


def open_database(ctx) -> Any:
    """DuckDB with the audited inputs (Colab), or an SQLite stand-in (offline test)."""

    if ctx.step2.get("connection") is not None:
        return ctx.step2["connection"]
    if ctx.step2.get("local_test"):
        connection = offline_test_database(ctx.step2["table_specs"], seed=7)
        ctx.step2["connection"] = connection
        ctx.step2["database_label"] = "sqlite-offline-stand-in"
        return connection
    import arbiter_data as data

    paths = ctx.step2["data_paths"]
    connection, provenance = data.prepare_duckdb(
        project_root=paths["local_root"],
        tpch_scale_factor=1.0,
        nyc_parquet_path=paths.get("nyc_parquet"),
        adult_zip_path=paths.get("adult_zip"),
    )
    ctx.step2["connection"] = connection
    ctx.step2["database_label"] = "duckdb"
    ctx.step2["database_provenance"] = provenance
    return connection


def offline_test_database(table_specs: Mapping[str, Any], *, seed: int, rows: int = 400) -> sqlite3.Connection:
    """Synthetic tables with the audited schemas, for offline code-path tests only."""

    rng = random.Random(seed)
    connection = sqlite3.connect(":memory:")
    for dataset, tables in table_specs.items():
        for spec in tables:
            table = str(spec["table"])
            columns = set(str(c) for c in spec.get("output_columns", []))
            columns |= set(str(c) for c in spec.get("measure_columns", []))
            columns |= set(str(c) for c in dict(spec.get("numeric_columns", {})))
            columns |= set(str(c) for c in dict(spec.get("categorical_columns", {})))
            categorical = {str(c): [str(v) for v in values] for c, values in dict(spec.get("categorical_columns", {})).items()}
            numeric = {str(c): [float(v) for v in values] for c, values in dict(spec.get("numeric_columns", {})).items()}
            ordered = sorted(columns)
            connection.execute(f'DROP TABLE IF EXISTS "{table}"')
            connection.execute(f'CREATE TABLE "{table}" (' + ", ".join(f'"{c}"' for c in ordered) + ")")
            data_rows = []
            for _ in range(rows):
                values = []
                for c in ordered:
                    if c in categorical and rng.random() < 0.95:
                        values.append(rng.choice(categorical[c]))
                    elif c in numeric and len(numeric[c]) >= 2:
                        values.append(rng.uniform(min(numeric[c]), max(numeric[c])) if rng.random() < 0.97 else None)
                    else:
                        values.append(round(rng.uniform(0, 100), 2) if rng.random() < 0.97 else None)
                data_rows.append(values)
            connection.executemany(f'INSERT INTO "{table}" VALUES (' + ", ".join("?" for _ in ordered) + ")", data_rows)
    return connection


def _fetch_scalar(connection, sql: str) -> Any:
    rows = connection.execute(sql).fetchall()
    if len(rows) != 1 or len(rows[0]) != 1:
        raise ValueError(f"expected one value from {sql[:80]}")
    return rows[0][0]


def _values_match(left: Any, right: Any, *, exact: bool) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if exact:
        return int(left) == int(right)
    left, right = float(left), float(right)
    return math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-8)


def _group_key(value: Any) -> Any:
    # NaN never equals itself, so it would break the per-group comparison.
    return "__nan__" if isinstance(value, float) and math.isnan(value) else value


def _grouped_rows(connection, sql: str) -> dict[Any, Any]:
    rows = connection.execute(sql).fetchall()
    result = {_group_key(row[0]): row[1] for row in rows}
    if len(result) != len(rows):
        raise ValueError("a grouped answer repeats a group key")
    return result


def _grouped_match(left: Mapping[Any, Any], right: Mapping[Any, Any], *, exact: bool) -> bool:
    return set(left) == set(right) and all(_values_match(left[key], right[key], exact=exact) for key in left)


def executed_check(connection, relation: Mapping[str, Any], descriptors: Mapping[str, wl.QueryDescriptor],
                   sql_by_id: Mapping[str, str], *, family: str) -> dict[str, Any]:
    """Execute one relation of the grouped extension on the data and compare both sides.

    The check composes the answers named by the relation in SQL, exactly as a
    buyer who holds those answers would, and compares the result with the
    answer of the priced query: per group for grouped answers, as one value
    otherwise. Counts must match exactly; sums to a relative 1e-10.
    """

    started = time.perf_counter_ns()
    detail = ""
    try:
        if family in DECOMPOSITION_FAMILIES:
            target = str(relation["target_id"])
            left, right = json.loads(str(relation["component_ids"]))
            d = descriptors[target]
            exact = d.aggregate == "count"
            union = f"{sql_by_id[left]} UNION ALL {sql_by_id[right]}"
            if d.mode == GROUPED_MODE:
                g = wl._quote_identifier(d.output_columns[0])
                target_rows = _grouped_rows(connection, sql_by_id[target])
                combined_rows = _grouped_rows(connection, f"SELECT u.{g}, SUM(u.value) AS value FROM ({union}) AS u GROUP BY u.{g}")
                passed = _grouped_match(target_rows, combined_rows, exact=exact)
                detail = f"{len(target_rows)} groups"
            else:
                target_value = _fetch_scalar(connection, sql_by_id[target])
                combined_value = _fetch_scalar(connection, f"SELECT SUM(u.value) AS value FROM ({union}) AS u")
                passed = _values_match(target_value, combined_value, exact=exact)
                detail = f"target {target_value} combined {combined_value}"
        else:
            derived, source = str(relation["derived_id"]), str(relation["source_id"])
            d, s = descriptors[derived], descriptors[source]
            exact = d.aggregate == "count"
            g = wl._quote_identifier(s.output_columns[0])
            values = ", ".join(wl._sql_literal(item) for item in d.categories)
            if family == "group_filter":
                derived_rows = _grouped_rows(connection, sql_by_id[derived])
                combined_rows = _grouped_rows(connection, f"SELECT u.{g}, u.value FROM ({sql_by_id[source]}) AS u WHERE u.{g} IN ({values})")
                passed = _grouped_match(derived_rows, combined_rows, exact=exact)
                detail = f"{len(derived_rows)} groups"
            else:
                if family == "group_rollup":
                    combined_sql = f"SELECT SUM(u.value) AS value FROM ({sql_by_id[source]}) AS u"
                elif family == "group_slice":
                    combined_sql = f"SELECT SUM(u.value) AS value FROM ({sql_by_id[source]}) AS u WHERE u.{g} IN ({values})"
                else:
                    raise ValueError(f"no executed check for family {family}")
                derived_value = _fetch_scalar(connection, sql_by_id[derived])
                combined_value = _fetch_scalar(connection, combined_sql)
                if exact and combined_value is None:
                    combined_value = 0  # a sum over no groups is the count of zero rows
                passed = _values_match(derived_value, combined_value, exact=exact)
                detail = f"derived {derived_value} combined {combined_value}"
        error = ""
    except Exception as exc:  # noqa: BLE001 - a failed check is a recorded failure
        passed, error = False, f"{type(exc).__name__}: {exc}"
    return {
        "relation_id": str(relation["relation_id"]),
        "proof_rule": str(relation["proof_rule"]),
        "passed": bool(passed),
        "detail": detail,
        "error": error,
        "runtime_ms": (time.perf_counter_ns() - started) / 1e6,
    }


# ---------------------------------------------------------------------------
# Solvers and methods
# ---------------------------------------------------------------------------


def clarabel_projection(snapshot: core.ConstraintSnapshot, candidate: np.ndarray, *, tolerance: float,
                        max_iter: int) -> tuple[np.ndarray, np.ndarray | None, str, int]:
    """Euclidean projection with Clarabel's interior-point method (not ADMM).

    Returns the primal vector, the dual vector z of the cone constraints (the
    multipliers of A x <= b, used only for the duality bound of stage s24),
    the solver status, and the iteration count.
    """

    import clarabel

    vector = np.asarray(candidate, dtype=np.float64)
    n = snapshot.assets
    P = sparse.identity(n, format="csc", dtype=np.float64)
    A = snapshot.matrix.tocsc().astype(np.float64)
    b = np.asarray(snapshot.bounds, dtype=np.float64).copy()
    # Clarabel solves A x + s = b with s in the cone, so the nonnegative cone
    # encodes A x <= b; P is the identity and q = -p0 for min 1/2 ||x - p0||^2.
    cones = [clarabel.NonnegativeConeT(int(snapshot.constraints))]
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_iter = int(max_iter)
    settings.tol_gap_abs = float(tolerance)
    settings.tol_gap_rel = float(tolerance)
    settings.tol_feas = float(tolerance)
    solver = clarabel.DefaultSolver(P, -vector, A, b, cones, settings)
    solution = solver.solve()
    status = str(solution.status)
    x = np.asarray(solution.x, dtype=np.float64)
    if x.shape != (n,) or not np.isfinite(x).all():
        raise RuntimeError(f"Clarabel returned no finite solution: {status}")
    z = np.asarray(getattr(solution, "z", []), dtype=np.float64)
    duals = z if z.shape == (int(snapshot.constraints),) and np.isfinite(z).all() else None
    return x, duals, status, int(getattr(solution, "iterations", -1))


def run_method_ext(ctx, snapshot: core.ConstraintSnapshot, candidate: np.ndarray, method: str, *,
                   tolerance: float) -> dict[str, Any]:
    """Notebook 07's run_method plus the Clarabel projections."""

    if method not in CLARABEL_METHODS:
        outcome = rev.run_method(snapshot, candidate, method, tolerance=tolerance, gate_options=ctx.gate_options())
        outcome["status"] = outcome.get("solver", "")
        outcome["_duals"] = None
        return outcome
    cfg = ctx.config
    solver_tolerance = float(cfg["clarabel_tight_tolerance"] if method == "clarabel_tight" else cfg["clarabel_default_tolerance"])
    vector = np.asarray(candidate, dtype=np.float64)
    started = time.perf_counter_ns()
    error, status, iterations, duals = "", "", -1, None
    try:
        prices, duals, status, iterations = clarabel_projection(snapshot, vector, tolerance=solver_tolerance,
                                                                max_iter=int(cfg["clarabel_max_iter"]))
    except Exception as exc:  # noqa: BLE001 - recorded
        prices, error = vector.copy(), f"{type(exc).__name__}: {exc}"
    verification = core.independent_verify(snapshot, prices, tolerance=tolerance)
    runtime_ms = (time.perf_counter_ns() - started) / 1e6
    return {
        "method": method,
        "runtime_ms": float(runtime_ms),
        "verified": bool(verification.passed and not error),
        "rejected": bool(not verification.passed or bool(error)),
        "unsafe_release": False,
        "fallback_used": False,
        "solver": "clarabel",
        "status": status,
        "sweeps": int(iterations),
        "projected_rows": 0,
        "l1_adjustment": float(np.abs(prices - vector).sum()),
        "violations_after": int(verification.violations),
        "max_residual_after": float(verification.max_residual),
        "error": error,
        "_prices": prices,
        "_duals": duals,
    }


def split_outcome(outcome: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray, np.ndarray | None]:
    """Separate the recorded fields of a method outcome from its price and dual vectors."""

    record = {key: value for key, value in outcome.items() if not key.startswith("_")}
    return record, outcome["_prices"], outcome.get("_duals")


def solver_self_test(ctx) -> pd.DataFrame:
    """Every method the stages use, on a three-asset projection with a known answer.

    The single decomposition row p_a - p_b - p_c <= 0 is violated by 0.7 at
    (1.2, 0.2, 0.3), so the Euclidean projection moves each price by 0.7/3
    along the row. Clarabel at its default tolerance may stop just outside
    1e-12, so only its accuracy is required here; the stages record its
    verification. (Any input works with the real solvers; this one also avoids
    the simulated iteration-limit case of the offline OSQP stand-in.)
    """

    relations = pd.DataFrame([{
        "dataset": "self-test", "relation_id": "self_test_row", "relation_type": "decomposition",
        "target_id": "a", "component_ids": json.dumps(["b", "c"]), "derived_id": "", "source_id": "",
        "sound": True, "proof_rule": "self_test",
    }])
    snapshot = core.build_snapshot(["a", "b", "c"], relations, epoch=1)
    candidate = np.array([1.2, 0.2, 0.3])
    expected = candidate - 0.7 * np.array([1.0, -1.0, -1.0]) / 3.0
    rows = []
    for method in ("arbiter_active", "full_cyclic", "osqp_qp", "clarabel_default", "clarabel_tight"):
        outcome, prices, _ = split_outcome(run_method_ext(ctx, snapshot, candidate, method, tolerance=PRIMARY_TOLERANCE))
        error = float(np.abs(prices - expected).max())
        accurate = error <= 1e-6 and not outcome.get("error")
        rows.append({"method": method, "verified_at_1e-12": bool(outcome["verified"]), "max_abs_error": error,
                     "status": str(outcome.get("status", "")), "error": str(outcome.get("error", "")),
                     "passed": bool(accurate and (outcome["verified"] or method in CLARABEL_METHODS))})
    return pd.DataFrame(rows)


def gate(ctx, snapshot, candidate, *, tolerance: float = PRIMARY_TOLERANCE):
    return rev.gate(snapshot, candidate, tolerance=tolerance, gate_options=ctx.gate_options())


def gate_record(ctx, snapshot, candidate, *, tolerance: float = PRIMARY_TOLERANCE) -> dict[str, Any]:
    before = core.independent_verify(snapshot, candidate, tolerance=tolerance)
    result = gate(ctx, snapshot, candidate, tolerance=tolerance)
    oracle = core.independent_verify(snapshot, result.prices, tolerance=tolerance)
    released = not result.rejected
    return {
        "violations_before": int(before.violations),
        "max_residual_before": float(before.max_residual),
        "released": bool(released),
        "verified": bool(result.verification.passed),
        "unsafe_release": bool(released and not oracle.passed),
        "fallback_used": bool(result.fallback_used),
        "solver": str(result.solver),
        "sweeps": int(result.sweeps),
        "runtime_ms": float(result.runtime_ms),
        "max_residual_after": float(oracle.max_residual),
        "l1_adjustment": float(result.l1_adjustment),
        "error": str(result.error),
        "_prices": result.prices,
    }


def row_box_attacks(snapshot: core.ConstraintSnapshot, center: np.ndarray, rows: Sequence[int], *,
                    budget_fraction: float, limit: int, tolerance: float) -> list[dict[str, Any]]:
    """core.exact_box_row_attacks restricted to a given set of rows (same construction)."""

    center = np.asarray(center, dtype=np.float64)
    if not core.independent_verify(snapshot, center, tolerance=tolerance).passed:
        raise ValueError("row_box_attacks requires a verified center")
    scale = max(float(np.median(np.abs(center))), tolerance)
    widths = float(budget_fraction) * np.maximum(np.abs(center), scale)
    lower = np.maximum(0.0, center - widths)
    upper = center + widths
    row_ids = snapshot.row_metadata["row_id"].astype(str).to_numpy()
    ranked = []
    for row in rows:
        start, end = snapshot.matrix.indptr[row], snapshot.matrix.indptr[row + 1]
        columns = snapshot.matrix.indices[start:end]
        coefficients = snapshot.matrix.data[start:end]
        extremal = np.where(coefficients >= 0, upper[columns], lower[columns])
        residual = float(np.dot(coefficients, extremal) - snapshot.bounds[row])
        ranked.append((residual, str(row_ids[row]), int(row)))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    attacks = []
    for residual, row_id, row in [item for item in ranked if item[0] > tolerance][: int(limit)]:
        start, end = snapshot.matrix.indptr[row], snapshot.matrix.indptr[row + 1]
        columns = snapshot.matrix.indices[start:end]
        coefficients = snapshot.matrix.data[start:end]
        candidate = center.copy()
        candidate[columns] = np.where(coefficients >= 0, upper[columns], lower[columns])
        attacks.append({"row_index": row, "row_id": row_id, "certified_target_residual": residual,
                        "budget_fraction": float(budget_fraction), "candidate": candidate})
    return attacks


# ---------------------------------------------------------------------------
# Stage s21: GROUP BY extension
# ---------------------------------------------------------------------------


def relation_family(record: Mapping[str, Any], construction: Mapping[str, wl.QueryDescriptor],
                    new_ids: set[str]) -> str:
    """Name the rule family of a relation from the construction descriptors of its assets."""

    if not (relation_assets(record) & new_ids):
        return f"base_{record['relation_type']}"
    if record["relation_type"] == "decomposition":
        target = construction[str(record["target_id"])]
        return "grouped_partition_union" if target.mode == GROUPED_MODE else "partition_union_new_assets"
    derived = construction[str(record["derived_id"])]
    source = construction[str(record["source_id"])]
    if source.mode == GROUPED_MODE:
        if derived.mode == GROUPED_MODE:
            return "group_filter"
        if derived.predicate_key() == source.predicate_key():
            return "group_rollup"
        return "group_slice"
    return f"new_{record['relation_type']}"


def _stable_sample(frame: pd.DataFrame, count: int, *salt: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    keys = frame["relation_id"].astype(str).map(lambda rid: rev.stable_rank_key(*salt, rid))
    return frame.assign(_key=keys).sort_values("_key").head(int(count)).drop(columns="_key")


_GROUPBY_NEGATIVE = re.compile(r'^SELECT "([^"]+)", COUNT\(\*\) AS n FROM "([^"]+)" GROUP BY "([^"]+)"$')


def negative_query_audit(dataset: str, negatives: pd.DataFrame) -> pd.DataFrame:
    """Re-parse the audited unsupported queries under the extended fragment.

    The audited suite includes "SELECT g, COUNT(*) AS n FROM t GROUP BY g" as a
    negative case. The extension supports exactly that shape, so those queries
    must now parse to the grouped descriptor; every other negative case must
    still be rejected.
    """

    rows = []
    for row in negatives[negatives["dataset"] == dataset].itertuples(index=False):
        sql = str(row.sql)
        match = _GROUPBY_NEGATIVE.match(sql)
        expected = None
        if match and match.group(1) == match.group(3):
            expected = wl.QueryDescriptor(table=match.group(2), mode=GROUPED_MODE, output_columns=(match.group(1),),
                                          aggregate="count", measure_column="*", predicate_kind="all",
                                          predicate_column="", lower=None, upper=None, categories=())
        try:
            parsed = descriptor_from_sql_ext(sql)
            supported, reason = True, parsed.to_json()
        except Exception as exc:  # noqa: BLE001 - the reason is the result
            parsed, supported, reason = None, False, f"{type(exc).__name__}: {exc}"
        if str(row.family) == "unsupported_groupby":
            passed = bool(expected is not None and supported and parsed == expected)
        else:
            passed = not supported
        rows.append({"dataset": dataset, "asset_id": str(row.asset_id), "family": str(row.family), "sql": sql,
                     "now_supported": supported, "expected_supported_under_extension": str(row.family) == "unsupported_groupby",
                     "passed": passed, "reason": reason})
    return pd.DataFrame(rows, columns=["dataset", "asset_id", "family", "sql", "now_supported",
                                       "expected_supported_under_extension", "passed", "reason"])


def s21_dataset(ctx, state) -> dict[str, pd.DataFrame]:
    cfg = ctx.config
    dataset = state.name
    specs = ctx.step2["table_specs"][dataset]
    base_assets = state.assets.copy()
    base_assets["asset_id"] = base_assets["asset_id"].astype(str)
    new_assets = generate_grouped_assets(dataset, specs, set(base_assets["asset_id"]), cfg)
    if new_assets.empty:
        raise RuntimeError(f"no grouped assets generated for {dataset}")
    all_assets = pd.concat([base_assets[ASSET_COLUMNS], new_assets[ASSET_COLUMNS]], ignore_index=True)
    order = all_assets["asset_id"].astype(str).tolist()
    new_ids = set(new_assets["asset_id"].astype(str))
    construction = {a: wl.QueryDescriptor.from_json(str(j)) for a, j in zip(order, all_assets["oracle_descriptor"])}

    # Compile: parse every asset with SQLGlot, then infer relations over base plus new assets.
    started = time.perf_counter_ns()
    descriptors: dict[str, wl.QueryDescriptor] = {}
    parse_rows = []
    for row in all_assets.itertuples(index=False):
        origin = "new" if str(row.asset_id) in new_ids else "base"
        try:
            parsed = descriptor_from_sql_ext(str(row.sql))
            descriptors[str(row.asset_id)] = parsed
            parse_rows.append({"dataset": dataset, "asset_id": str(row.asset_id), "origin": origin,
                               "parsed_supported": True,
                               "matches_construction": bool(parsed == construction[str(row.asset_id)]), "reason": ""})
        except Exception as exc:  # noqa: BLE001
            parse_rows.append({"dataset": dataset, "asset_id": str(row.asset_id), "origin": origin,
                               "parsed_supported": False, "matches_construction": False,
                               "reason": f"{type(exc).__name__}: {exc}"})
    parse_ms = (time.perf_counter_ns() - started) / 1e6
    parse_audit = pd.DataFrame(parse_rows)
    started = time.perf_counter_ns()
    compiled = infer_relations_ext(all_assets, descriptors)
    infer_ms = (time.perf_counter_ns() - started) / 1e6
    compiled_records = compiled.to_dict(orient="records")
    family_by_id = {str(r["relation_id"]): relation_family(r, construction, new_ids) for r in compiled_records}
    base_compiled = [r for r in compiled_records if family_by_id[str(r["relation_id"])].startswith("base_")]
    new_compiled = [r for r in compiled_records if not family_by_id[str(r["relation_id"])].startswith("base_")]
    audited = ctx.step2["audited_relations"]
    audited_ids = set(audited[audited["dataset"] == dataset]["relation_id"].astype(str))
    base_unchanged = {str(r["relation_id"]) for r in base_compiled} == audited_ids

    # Oracle: the audited pairwise decomposition oracle (mode-generic) plus the
    # independent pairwise oracle for the grouped rules, on relations that touch new assets.
    oracle = pd.concat([wl.oracle_relations(all_assets), oracle_grouped_derivations(all_assets)], ignore_index=True)
    oracle_records = [r for r in oracle.to_dict(orient="records") if relation_assets(r) & new_ids]
    found = {relation_key(r): r for r in new_compiled}
    truth = {relation_key(r): r for r in oracle_records}
    metric_rows = []
    for family in NEW_FAMILIES:
        f_keys = {k for k, r in found.items() if relation_family(r, construction, new_ids) == family}
        t_keys = {k for k, r in truth.items() if relation_family(r, construction, new_ids) == family}
        tp, fp, fn = len(f_keys & t_keys), len(f_keys - t_keys), len(t_keys - f_keys)
        metric_rows.append({"dataset": dataset, "family": family, "oracle_relations": len(t_keys),
                            "compiled_relations": len(f_keys), "true_positives": tp, "false_positives": fp,
                            "false_negatives": fn, "precision": tp / (tp + fp) if tp + fp else 1.0,
                            "recall": tp / (tp + fn) if tp + fn else 1.0})
    other = {relation_family(r, construction, new_ids) for r in list(found.values()) + list(truth.values())} - set(NEW_FAMILIES)
    compiler_metrics = pd.DataFrame(metric_rows)

    # Executed checks on real data.
    connection = open_database(ctx)
    sql_by_id = dict(zip(order, all_assets["sql"].astype(str)))
    new_frame = pd.DataFrame(new_compiled, columns=RELATION_COLUMNS)
    new_frame["family"] = [family_by_id[str(r["relation_id"])] for r in new_compiled]
    exec_rows = []
    for family, group in new_frame.groupby("family", sort=True):
        for record in _stable_sample(group, int(cfg["groupby_exec_checks_per_family"]), dataset, family).to_dict(orient="records"):
            outcome = executed_check(connection, record, construction, sql_by_id, family=family)
            exec_rows.append({"dataset": dataset, "family": family, **outcome})
    exec_checks = pd.DataFrame(exec_rows)

    # Features and controlled prices with the audited formula and the base scales.
    # The audited plan-feature step recorded estimated_rows = 0 for every asset
    # (its pattern does not match the EXPLAIN text of DuckDB 1.3.2), so the new
    # assets carry the same constant and the feature distribution stays aligned.
    new_features = ex.add_structural_features(new_assets.assign(estimated_rows=0.0))
    base_features = state.features.copy()
    interval_scale, category_scale = base_price_scales(base_features)
    base_reproduced = controlled_policy_fixed_scale(base_features, interval_scale, category_scale)
    features = pd.concat([base_features, new_features[list(base_features.columns)]], ignore_index=True, sort=False)
    features = features.set_index("asset_id").loc[order].reset_index()
    list_prices = controlled_policy_fixed_scale(features, interval_scale, category_scale)
    snapshot = core.build_snapshot(order, compiled[RELATION_COLUMNS], epoch=1)
    policy_check = core.independent_verify(snapshot, list_prices, tolerance=PRIMARY_TOLERANCE)
    meta = snapshot.row_metadata
    meta_ids = meta["row_id"].astype(str).to_numpy()
    row_family = np.asarray([family_by_id.get(rid, str(kind)) for rid, kind in zip(meta_ids, meta["constraint_type"])])
    new_row_mask = np.isin(row_family, NEW_FAMILIES)

    # Proposals from the three model families, retrained over the extended workload.
    targets = pd.DataFrame({"dataset": dataset, "asset_id": order, "target_price": list_prices})
    _, predictions, pooled = ex.grouped_oof_predictions(
        features, targets, seeds=[int(s) for s in cfg["seeds"]], model_names=rev.MODEL_NAMES
    )
    gate_rows = []
    for (model, seed), group in predictions.groupby(["model", "seed"], sort=True):
        ctx.beat("s21 proposals")
        proposal = group.set_index("asset_id")["oof_prediction"].loc[order].to_numpy(np.float64)
        residuals = np.asarray(snapshot.matrix @ proposal - snapshot.bounds).ravel()
        outcome = gate_record(ctx, snapshot, proposal)
        outcome.pop("_prices")
        gate_rows.append({"dataset": dataset, "model": model, "seed": int(seed),
                          "new_family_rows_violated_before": int((residuals[new_row_mask] > PRIMARY_TOLERANCE).sum()),
                          **outcome})
    proposals_gate = pd.DataFrame(gate_rows)

    # Per-row attacks on each new family, with the audited attack centers.
    attack_rows = []
    derivation_center, derivation_label = ex.relation_family_attack_center(
        snapshot, list_prices, "projection_derivation", tolerance=PRIMARY_TOLERANCE)
    for family in NEW_FAMILIES:
        rows = np.flatnonzero(row_family == family).tolist()
        if not rows:
            continue
        center = list_prices if family in DECOMPOSITION_FAMILIES else derivation_center
        for budget in cfg["groupby_attack_budgets"]:
            for attack in row_box_attacks(snapshot, center, rows, budget_fraction=float(budget),
                                          limit=int(cfg["groupby_attack_cases"]), tolerance=PRIMARY_TOLERANCE):
                outcome = gate_record(ctx, snapshot, attack["candidate"])
                outcome.pop("_prices")
                attack_rows.append({"dataset": dataset, "family": family, "row_id": attack["row_id"],
                                    "budget": float(budget), "certified_target_residual": attack["certified_target_residual"],
                                    **outcome})
    attacks = pd.DataFrame(attack_rows)

    negatives = negative_query_audit(dataset, ctx.step2["negative_assets"])
    family_counts = pd.Series(new_frame["family"]).value_counts()
    family_rows = pd.Series(row_family[new_row_mask]).value_counts()
    base_parse = parse_audit[parse_audit["origin"] == "base"]
    new_parse = parse_audit[parse_audit["origin"] == "new"]
    summary = pd.DataFrame([{
        "dataset": dataset,
        "base_assets": len(base_assets),
        "new_assets": len(new_assets),
        "new_grouped_assets": int(sum(construction[a].mode == GROUPED_MODE for a in new_ids)),
        "new_relations": len(new_compiled),
        **{f"relations_{family}": int(family_counts.get(family, 0)) for family in NEW_FAMILIES},
        **{f"rows_{family}": int(family_rows.get(family, 0)) for family in NEW_FAMILIES},
        "unexpected_families": ";".join(sorted(other)),
        "base_relations_unchanged": bool(base_unchanged),
        "base_parse_matches_construction": bool(base_parse["matches_construction"].all()),
        "new_parse_all_supported": bool(new_parse["parsed_supported"].all()),
        "new_parse_matches_construction": bool(new_parse["matches_construction"].all()),
        "compiler_precision_min": float(compiler_metrics["precision"].min()),
        "compiler_recall_min": float(compiler_metrics["recall"].min()),
        "executed_checks": len(exec_checks),
        "executed_checks_passed": int(exec_checks["passed"].sum()) if len(exec_checks) else 0,
        "negatives": len(negatives),
        "negatives_now_supported": int(negatives["now_supported"].sum()) if len(negatives) else 0,
        "negatives_passed": int(negatives["passed"].sum()) if len(negatives) else 0,
        "list_prices_verify": bool(policy_check.passed),
        "list_prices_max_residual": float(policy_check.max_residual),
        "base_prices_reproduced": bool(np.array_equal(base_reproduced, state.list_prices)),
        "snapshot_rows": snapshot.constraints,
        "snapshot_nonzeros": snapshot.nonzeros,
        "parse_ms": parse_ms,
        "infer_ms": infer_ms,
        "proposals": len(proposals_gate),
        "proposals_released": int(proposals_gate["released"].sum()),
        "proposals_unsafe": int(proposals_gate["unsafe_release"].sum()),
        "proposals_fallback": int(proposals_gate["fallback_used"].sum()),
        "proposal_violations_before_min": int(proposals_gate["violations_before"].min()),
        "proposal_violations_before_max": int(proposals_gate["violations_before"].max()),
        "proposal_new_family_rows_violated_max": int(proposals_gate["new_family_rows_violated_before"].max()),
        "proposal_sweeps_max": int(proposals_gate["sweeps"].max()),
        "proposal_gate_p50_ms": float(proposals_gate["runtime_ms"].median()),
        "attacks": len(attacks),
        "attacks_released": int(attacks["released"].sum()) if len(attacks) else 0,
        "attacks_unsafe": int(attacks["unsafe_release"].sum()) if len(attacks) else 0,
        "attacks_fallback": int(attacks["fallback_used"].sum()) if len(attacks) else 0,
        "attack_sweeps_max": int(attacks["sweeps"].max()) if len(attacks) else 0,
        "attack_gate_p50_ms": float(attacks["runtime_ms"].median()) if len(attacks) else math.nan,
        "attack_max_residual_after": float(attacks["max_residual_after"].max()) if len(attacks) else math.nan,
        "database": ctx.step2.get("database_label", ""),
        "derivation_attack_center": derivation_label,
    }])
    return {
        "summary": summary,
        "new_assets": new_assets,
        "new_relations": new_frame,
        "parse_audit_new": new_parse.reset_index(drop=True),
        "parse_audit_base_mismatches": base_parse[~base_parse["matches_construction"]].reset_index(drop=True),
        "negative_audit": negatives,
        "compiler_metrics": compiler_metrics,
        "executed_checks": exec_checks,
        "proposals_gate": proposals_gate,
        "model_pooled": pooled.assign(stage="s21"),
        "attacks": attacks,
    }


def run_stage_21(ctx) -> dict[str, pd.DataFrame]:
    parts = [(f"{state.key}/groupby", lambda state=state: s21_dataset(ctx, state)) for state in ctx.datasets.values()]
    frames = rev.run_parts(ctx, "s21", parts)
    frames["parser_self_test"] = parser_self_test()
    rev.write_outputs(ctx, "s21", frames)
    summary = frames["summary"]
    ctx.add_check("s21", "every dataset produced a grouped workload", len(summary) == len(ctx.datasets))
    ctx.add_check("s21", "extended parser self-test (supported and fail-closed strings)",
                  bool(_as_flag(frames["parser_self_test"]["passed"]).all()))
    ctx.add_check("s21", "base relations are unchanged by the extension",
                  bool(_as_flag(summary["base_relations_unchanged"]).all()))
    ctx.add_check("s21", "base assets parse exactly as in the audited compile",
                  bool(_as_flag(summary["base_parse_matches_construction"]).all()))
    ctx.add_check("s21", "every new asset parses to its construction descriptor",
                  bool(_as_flag(summary["new_parse_all_supported"]).all()
                       and _as_flag(summary["new_parse_matches_construction"]).all()))
    ctx.add_check("s21", "no relation outside the named families touches a new asset",
                  bool((summary["unexpected_families"].fillna("").astype(str) == "").all()))
    metrics = frames["compiler_metrics"]
    ctx.add_check("s21", "compiler matches the oracle on every new relation family (precision and recall 1)",
                  bool(len(metrics) and (metrics["precision"] == 1.0).all() and (metrics["recall"] == 1.0).all()),
                  detail=metrics[["dataset", "family", "precision", "recall"]].to_json(orient="records"))
    ctx.add_check("s21", "every new relation family occurs on every dataset",
                  bool(all((summary[f"relations_{family}"] > 0).all() for family in NEW_FAMILIES)), severity="finding")
    checks = frames["executed_checks"]
    ctx.add_check("s21", "every executed check passes on real data",
                  bool(len(checks) and _as_flag(checks["passed"]).all()),
                  detail=f"{int(_as_flag(checks['passed']).sum()) if len(checks) else 0}/{len(checks)}")
    negatives = frames["negative_audit"]
    ctx.add_check("s21", "audited GROUP BY negatives become supported; every other negative stays rejected",
                  bool(len(negatives) and _as_flag(negatives["passed"]).all()),
                  detail=f"{int(_as_flag(negatives['now_supported']).sum()) if len(negatives) else 0} now supported")
    ctx.add_check("s21", "extended list prices verify at 1e-12", bool(_as_flag(summary["list_prices_verify"]).all()))
    ctx.add_check("s21", "base list prices are reproduced exactly", bool(_as_flag(summary["base_prices_reproduced"]).all()))
    ctx.add_check("s21", "no proposal or attack is released unverified",
                  int(summary["proposals_unsafe"].sum()) == 0 and int(summary["attacks_unsafe"].sum()) == 0)
    expected_proposals = len(ctx.datasets) * len(rev.MODEL_NAMES) * len(ctx.config["seeds"])
    ctx.add_check("s21", "the proposal grid is complete", int(summary["proposals"].sum()) == expected_proposals,
                  detail=f"{int(summary['proposals'].sum())}/{expected_proposals}")
    return frames


# ---------------------------------------------------------------------------
# Stage s22: multi-part partition audit
# ---------------------------------------------------------------------------


def interval_surfaces(order: Sequence[str], descriptors: Mapping[str, wl.QueryDescriptor]) -> dict[tuple, list[tuple[int, float, float]]]:
    surfaces: dict[tuple, list[tuple[int, float, float]]] = defaultdict(list)
    for index, asset_id in enumerate(order):
        d = descriptors.get(str(asset_id))
        if d is None or d.predicate_kind != "numeric_interval":
            continue
        surfaces[d.relation_group_key()].append((index, float(d.lower), float(d.upper)))
    return dict(surfaces)


def cheapest_chain(intervals: Sequence[tuple[int, float, float]], exclude: int, a: float, b: float,
                   prices: np.ndarray) -> tuple[float, list[int]]:
    """Shortest path over contiguous same-surface intervals that tile [a, b)."""

    starts: dict[float, list[tuple[float, int]]] = defaultdict(list)
    nodes = {a, b}
    for index, lower, upper in intervals:
        if index == exclude or lower < a or upper > b or not lower < upper:
            continue
        starts[lower].append((upper, index))
        nodes.add(lower)
        nodes.add(upper)
    distance = {a: 0.0}
    previous: dict[float, tuple[float, int]] = {}
    for node in sorted(nodes):
        if node not in distance:
            continue
        for upper, index in starts.get(node, []):
            cost = distance[node] + float(prices[index])
            if cost < distance.get(upper, math.inf):
                distance[upper] = cost
                previous[upper] = (node, index)
    if b not in distance:
        return math.inf, []
    path, node = [], b
    while node != a:
        node, index = previous[node]
        path.append(index)
    return distance[b], path[::-1]


def multipart_audit(surfaces: Mapping[tuple, Sequence[tuple[int, float, float]]], prices: np.ndarray,
                    list_prices: np.ndarray, *, tolerance: float, targets: set[int] | None = None) -> pd.DataFrame:
    rows = []
    for key, intervals in surfaces.items():
        for index, lower, upper in intervals:
            if targets is not None and index not in targets:
                continue
            cost, path = cheapest_chain(intervals, index, lower, upper, prices)
            if not path:
                continue
            violation = float(prices[index]) - cost
            parts = len(path)
            rows.append({"target_index": int(index), "parts": parts, "chain_cost": cost,
                         "posted_price": float(prices[index]), "violation": violation,
                         "flagged": bool(violation > parts * tolerance),
                         "saving_vs_list": max(violation, 0.0) / float(list_prices[index]),
                         "path": json.dumps([int(i) for i in path])})
    return pd.DataFrame(rows, columns=["target_index", "parts", "chain_cost", "posted_price", "violation",
                                       "flagged", "saving_vs_list", "path"])


def _splits(lower: float, upper: float, occupied: set[tuple[float, float]], values: Sequence[float]) -> bool:
    """True when [lower, upper) is the union of two intervals in ``occupied``."""

    return any((lower, v) in occupied and (v, upper) in occupied for v in values if lower < v < upper)


def wide_targets(order: Sequence[str], descriptors: Mapping[str, wl.QueryDescriptor], specs: Sequence[Mapping[str, Any]],
                 cfg: Mapping[str, Any]) -> list[wl.QueryDescriptor]:
    """New interval assets that only chains of three or more existing parts can rebuild.

    For each predicate surface (same table, mode, outputs, aggregate, measure,
    and column), a candidate spans ``multipart_widths`` boundary gaps and is
    admitted only if (i) it is not an asset already, (ii) no two assets of the
    surface tile it, so rule R1 emits no row for it, (iii) it does not complete
    a two-part split of an already admitted target, and (iv) a chain of
    existing parts tiles it. Candidates are taken from the surfaces in turn.
    """

    boundaries = {}
    for spec in specs:
        for column, values in dict(spec.get("numeric_columns", {})).items():
            boundaries[(str(spec["table"]), str(column))] = sorted(set(float(v) for v in values if np.isfinite(v)))
    surfaces = interval_surfaces(order, descriptors)
    unit_prices = np.ones(len(order))
    pools: dict[tuple, list[tuple[float, float]]] = {}
    templates: dict[tuple, wl.QueryDescriptor] = {}
    for key in sorted(surfaces, key=str):
        intervals = surfaces[key]
        template = descriptors[str(order[intervals[0][0]])]
        values = boundaries.get((template.table, template.predicate_column), [])
        present = {(lo, hi) for _, lo, hi in intervals}
        pool = []
        for width in cfg["multipart_widths"]:
            for start in range(0, max(len(values) - int(width), 0)):
                lower, upper = values[start], values[start + int(width)]
                if (lower, upper) in present or _splits(lower, upper, present, values):
                    continue
                _, path = cheapest_chain(intervals, -1, lower, upper, unit_prices)
                if len(path) >= 3:
                    pool.append((lower, upper))
        if pool:
            pools[key] = pool
            templates[key] = template
    limit = int(cfg["multipart_targets_per_dataset"])
    admitted: dict[tuple, list[tuple[float, float]]] = defaultdict(list)
    chosen: list[wl.QueryDescriptor] = []
    keys = sorted(pools, key=str)
    while len(chosen) < limit and any(pools[k] for k in keys):
        for key in keys:
            if len(chosen) >= limit:
                break
            values = boundaries[(templates[key].table, templates[key].predicate_column)]
            present = {(lo, hi) for _, lo, hi in surfaces[key]}
            while pools[key]:
                lower, upper = pools[key].pop(0)
                occupied = present | set(admitted[key]) | {(lower, upper)}
                if _splits(lower, upper, occupied - {(lower, upper)}, values):
                    continue
                if any(_splits(lo, hi, occupied - {(lo, hi)}, values) for lo, hi in admitted[key]):
                    continue
                admitted[key].append((lower, upper))
                chosen.append(replace(templates[key], lower=lower, upper=upper))
                break
    return chosen


def s22_dataset(ctx, state) -> dict[str, pd.DataFrame]:
    cfg = ctx.config
    dataset = state.name
    base_assets = state.assets.copy()
    base_assets["asset_id"] = base_assets["asset_id"].astype(str)
    order = base_assets["asset_id"].tolist()
    descriptors = {a: wl.QueryDescriptor.from_json(str(j)) for a, j in zip(order, base_assets["oracle_descriptor"])}
    targets = wide_targets(order, descriptors, ctx.step2["table_specs"][dataset], cfg)
    if not targets:
        raise RuntimeError(f"no multi-part targets for {dataset}")
    wide_rows = []
    for descriptor in targets:
        sql = " ".join(wl.descriptor_to_sql(descriptor).split())
        wide_rows.append({"dataset": dataset, "asset_id": wl._stable_id("asset", dataset, sql), "sql": sql,
                          "family": "multipart_wide_target", "template_group": "multipart",
                          "expected_supported": True, "oracle_descriptor": descriptor.to_json()})
    wide_assets = pd.DataFrame(wide_rows, columns=ASSET_COLUMNS).drop_duplicates("asset_id")
    all_assets = pd.concat([base_assets[ASSET_COLUMNS], wide_assets], ignore_index=True)
    all_order = all_assets["asset_id"].astype(str).tolist()
    parsed = {a: descriptor_from_sql_ext(s) for a, s in zip(all_order, all_assets["sql"].astype(str))}
    construction = {a: wl.QueryDescriptor.from_json(str(j)) for a, j in zip(all_order, all_assets["oracle_descriptor"])}
    parse_matches = all(parsed[a] == construction[a] for a in all_order)
    relations = wl.infer_relations_from_descriptors(all_assets, parsed)
    wide_ids = set(wide_assets["asset_id"].astype(str))
    r1_on_wide = int(((relations["relation_type"] == "decomposition") & relations["target_id"].astype(str).isin(wide_ids)).sum())
    snapshot = core.build_snapshot(all_order, relations, epoch=1)
    features = ex.add_structural_features(all_assets.assign(estimated_rows=0.0))
    interval_scale, category_scale = base_price_scales(state.features)
    list_prices = controlled_policy_fixed_scale(features, interval_scale, category_scale)
    base_reproduced = bool(np.array_equal(list_prices[: len(order)], state.list_prices))
    policy_ok = core.independent_verify(snapshot, list_prices, tolerance=PRIMARY_TOLERANCE).passed
    surfaces = interval_surfaces(all_order, parsed)
    wide_index = {i for i, a in enumerate(all_order) if a in wide_ids}

    # Audit on list prices and its cost.
    list_audit = multipart_audit(surfaces, list_prices, list_prices, tolerance=PRIMARY_TOLERANCE)
    audit_ms = []
    for _ in range(int(cfg["audit_repetitions"])):
        started = time.perf_counter_ns()
        multipart_audit(surfaces, list_prices, list_prices, tolerance=PRIMARY_TOLERANCE)
        audit_ms.append((time.perf_counter_ns() - started) / 1e6)

    # Attacks on wide targets: raise the target price, gate, audit, enforce lazily.
    kappa = float(np.median(np.abs(list_prices)))
    attack_rows = []
    for t in sorted(wide_index):
        ctx.beat(f"s22 {state.key}")
        for budget in cfg["multipart_attack_budgets"]:
            candidate = list_prices.copy()
            candidate[t] += float(budget) * max(abs(float(list_prices[t])), kappa)
            current = snapshot
            outcome = gate_record(ctx, current, candidate)
            first = multipart_audit(surfaces, outcome["_prices"], list_prices, tolerance=PRIMARY_TOLERANCE, targets={t})
            detected = bool(len(first) and first["flagged"].any())
            saving_before = float(first["saving_vs_list"].max()) if len(first) else math.nan
            rounds, rows_added, total_ms = 0, 0, outcome["runtime_ms"]
            final = outcome
            audit = first
            while bool(len(audit) and audit["flagged"].any()) and rounds < int(cfg["multipart_max_rounds"]):
                rounds += 1
                started = time.perf_counter_ns()
                new_rows = []
                for record in audit[audit["flagged"]].to_dict(orient="records"):
                    parts = [all_order[i] for i in json.loads(record["path"])]
                    target_id = all_order[int(record["target_index"])]
                    new_rows.append({"dataset": dataset,
                                     "relation_id": wl._stable_id("multipart", dataset, target_id, *sorted(parts)),
                                     "relation_type": "decomposition", "target_id": target_id,
                                     "component_ids": json.dumps(sorted(parts)), "derived_id": "", "source_id": "",
                                     "sound": True, "proof_rule": "multipart_partition_union"})
                existing_ids = set(current.row_metadata["row_id"].astype(str))
                new_rows = [r for r in new_rows if r["relation_id"] not in existing_ids]
                if not new_rows:
                    break
                current = core.extend_snapshot(current, pd.DataFrame(new_rows), new_epoch=current.epoch + 1)
                rows_added += len(new_rows)
                extend_ms = (time.perf_counter_ns() - started) / 1e6
                final = gate_record(ctx, current, candidate)
                audit = multipart_audit(surfaces, final["_prices"], list_prices, tolerance=PRIMARY_TOLERANCE, targets={t})
                total_ms += extend_ms + final["runtime_ms"]
            attack_rows.append({
                "dataset": dataset, "target_asset_id": all_order[t], "budget": float(budget),
                "chain_parts": int(first["parts"].iloc[0]) if len(first) else 0,
                "released_first": bool(outcome["released"]), "unsafe_first": bool(outcome["unsafe_release"]),
                "audit_detected": detected, "saving_before_vs_list": saving_before,
                "enforcement_rounds": rounds, "rows_added": rows_added,
                "released_final": bool(final["released"]), "unsafe_final": bool(final["unsafe_release"]),
                "flagged_after": bool(len(audit) and audit["flagged"].any()),
                "saving_after_vs_list": float(audit["saving_vs_list"].max()) if len(audit) else math.nan,
                "first_gate_ms": float(outcome["runtime_ms"]), "total_ms": float(total_ms),
            })
    attacks = pd.DataFrame(attack_rows)

    # Audit mode: check releases under partial declarations against the complete snapshot.
    complete = state.snapshot
    meta = complete.row_metadata
    types = meta["constraint_type"].astype(str).to_numpy()
    family_code = {"decomposition": "D", "projection_derivation": "P", "filter_derivation": "F"}
    audit_rows = []
    proposals = state.ordered_proposals()[: int(cfg["audit_mode_proposals"])]
    for declaration in cfg["audit_mode_declarations"]:
        omit = [i for i, kind in enumerate(types) if kind in family_code and family_code[kind] not in declaration]
        partial = core.snapshot_without_rows(complete, omit, new_epoch=complete.epoch + 1)
        for (model, seed), proposal in proposals:
            outcome = gate_record(ctx, partial, proposal)
            times = []
            check = None
            for _ in range(int(cfg["audit_repetitions"])):
                started = time.perf_counter_ns()
                check = core.independent_verify(complete, outcome["_prices"], tolerance=PRIMARY_TOLERANCE)
                times.append((time.perf_counter_ns() - started) / 1e6)
            audit_rows.append({"dataset": dataset, "declaration": declaration, "model": model, "seed": int(seed),
                               "released_partial": bool(outcome["released"]), "gate_ms": float(outcome["runtime_ms"]),
                               "audit_flags_undeclared_violation": bool(not check.passed),
                               "undeclared_rows_violated": int(check.violations),
                               "audit_p50_ms": float(np.median(times))})
    audit_mode = pd.DataFrame(audit_rows)
    summary = pd.DataFrame([{
        "dataset": dataset,
        "wide_targets": len(wide_ids),
        "parse_matches_construction": bool(parse_matches),
        "r1_rows_on_wide_targets": r1_on_wide,
        "base_prices_reproduced": base_reproduced,
        "list_prices_verify": bool(policy_ok),
        "interval_targets_audited": len(list_audit),
        "list_price_flags": int(list_audit["flagged"].sum()) if len(list_audit) else 0,
        "audit_all_targets_p50_ms": float(np.median(audit_ms)),
        "audit_all_targets_p95_ms": rev.q95(audit_ms),
        "chain_parts_median": float(attacks["chain_parts"].median()) if len(attacks) else math.nan,
        "chain_parts_max": int(attacks["chain_parts"].max()) if len(attacks) else 0,
        "attacks": len(attacks),
        "released_first": int(attacks["released_first"].sum()) if len(attacks) else 0,
        "audit_detected": int(attacks["audit_detected"].sum()) if len(attacks) else 0,
        "saving_before_max": float(attacks["saving_before_vs_list"].max()) if len(attacks) else math.nan,
        "enforcement_rounds_max": int(attacks["enforcement_rounds"].max()) if len(attacks) else 0,
        "flagged_after": int(attacks["flagged_after"].sum()) if len(attacks) else 0,
        "unsafe_any": int(attacks["unsafe_first"].sum() + attacks["unsafe_final"].sum()) if len(attacks) else 0,
        "enforced_total_p50_ms": float(attacks["total_ms"].median()) if len(attacks) else math.nan,
        "audit_mode_releases": len(audit_mode),
        "audit_mode_flagged": int(audit_mode["audit_flags_undeclared_violation"].sum()) if len(audit_mode) else 0,
        "audit_mode_p50_ms": float(audit_mode["audit_p50_ms"].median()) if len(audit_mode) else math.nan,
        "complete_snapshot_rows": complete.constraints,
    }])
    return {"summary": summary, "attacks": attacks, "audit_mode": audit_mode,
            "wide_assets": wide_assets, "list_audit": list_audit.assign(dataset=dataset)}


def run_stage_22(ctx) -> dict[str, pd.DataFrame]:
    parts = [(f"{state.key}/multipart", lambda state=state: s22_dataset(ctx, state)) for state in ctx.datasets.values()]
    frames = rev.run_parts(ctx, "s22", parts)
    rev.write_outputs(ctx, "s22", frames)
    summary = frames["summary"]
    ctx.add_check("s22", "every dataset produced multi-part targets", len(summary) == len(ctx.datasets))
    ctx.add_check("s22", "rule R1 emits no row for a multi-part target", int(summary["r1_rows_on_wide_targets"].sum()) == 0)
    ctx.add_check("s22", "list prices with multi-part targets verify, base prices unchanged",
                  bool(_as_flag(summary["list_prices_verify"]).all() and _as_flag(summary["base_prices_reproduced"]).all()))
    ctx.add_check("s22", "the audit flags nothing on list prices", int(summary["list_price_flags"].sum()) == 0)
    ctx.add_check("s22", "no release is unverified against its own snapshot", int(summary["unsafe_any"].sum()) == 0)
    ctx.add_check("s22", "lazy enforcement removes every flagged chain", int(summary["flagged_after"].sum()) == 0,
                  severity="finding")
    return frames


# ---------------------------------------------------------------------------
# Stage s23: price floors
# ---------------------------------------------------------------------------


def s23_cell(ctx, state, floor: float) -> dict[str, pd.DataFrame]:
    cfg = ctx.config
    floored = np.maximum(state.list_prices, float(floor))
    check = core.independent_verify(state.snapshot, floored, tolerance=PRIMARY_TOLERANCE)
    evaluator = rev.RelationEvaluator(state.snapshot, floored)
    target_min = float(evaluator.target_list.min())
    targets = pd.DataFrame({"dataset": state.name, "asset_id": state.asset_ids, "target_price": floored})
    metrics, predictions, pooled = ex.grouped_oof_predictions(
        state.features, targets, seeds=[int(s) for s in cfg["seeds"]], model_names=rev.MODEL_NAMES)
    rows = []
    for (model, seed), group in predictions.groupby(["model", "seed"], sort=True):
        proposal = group.set_index("asset_id")["oof_prediction"].loc[state.asset_ids].to_numpy(np.float64)
        for tolerance in cfg["floor_tolerances"]:
            outcome = gate_record(ctx, state.snapshot, proposal, tolerance=float(tolerance))
            metrics_row = evaluator.metrics(outcome["_prices"]) if outcome["released"] else {}
            outcome.pop("_prices")
            rows.append({"dataset": state.name, "floor": float(floor), "tolerance": float(tolerance),
                         "model": model, "seed": int(seed), "target_list_min": target_min,
                         "corollary_bound": float(tolerance) / target_min, **outcome, **metrics_row})
    gate_frame = pd.DataFrame(rows)
    policy = pd.DataFrame([{"dataset": state.name, "floor": float(floor), "verifies": bool(check.passed),
                            "max_residual": float(check.max_residual), "target_list_min": target_min,
                            "assets_floored": int((state.list_prices < float(floor)).sum()),
                            "target_list_min_usd": target_min * DOLLARS_PER_PRICE_UNIT}])
    return {"gate": gate_frame, "policy": policy, "model_pooled": pooled.assign(floor=float(floor))}


def run_stage_23(ctx) -> dict[str, pd.DataFrame]:
    cfg = ctx.config
    parts = [(f"{state.key}/floor-{floor:g}", lambda state=state, floor=floor: s23_cell(ctx, state, float(floor)))
             for state in ctx.datasets.values() for floor in cfg["price_floors"]]
    frames = rev.run_parts(ctx, "s23", parts)
    gate_frame = frames["gate"]
    released = gate_frame[_as_flag(gate_frame["released"])]
    summary = (
        gate_frame.assign(released_bool=_as_flag(gate_frame["released"]))
        .groupby(["floor", "tolerance"], as_index=False)
        .agg(gate_calls=("released_bool", "size"), released=("released_bool", "sum"),
             fallback=("fallback_used", lambda s: int(_as_flag(s).sum())),
             target_list_min=("target_list_min", "min"), corollary_bound=("corollary_bound", "max"),
             sweeps_max=("sweeps", "max"), gate_p50_ms=("runtime_ms", "median"))
    )
    if len(released):
        by = released.groupby(["floor", "tolerance"])
        summary = summary.merge(
            by.agg(max_saving_vs_list=("max_saving_vs_list", "max"),
                   releases_gt_1pct=("n_saving_gt_1pct", lambda s: int((s > 0).sum())),
                   releases_gt_0p1pct=("n_saving_gt_0p1pct", lambda s: int((s > 0).sum()))).reset_index(),
            on=["floor", "tolerance"], how="left")
    summary["bound_holds"] = summary["max_saving_vs_list"] <= summary["corollary_bound"] * (1 + 1e-9)
    summary["tolerance_usd"] = summary["tolerance"] * DOLLARS_PER_PRICE_UNIT
    summary["target_list_min_usd"] = summary["target_list_min"] * DOLLARS_PER_PRICE_UNIT
    frames["summary"] = summary
    rev.write_outputs(ctx, "s23", frames)
    expected = len(ctx.datasets) * len(cfg["price_floors"]) * len(cfg["floor_tolerances"]) * len(rev.MODEL_NAMES) * len(cfg["seeds"])
    ctx.add_check("s23", "the floor grid is complete", len(gate_frame) == expected, detail=f"{len(gate_frame)}/{expected}")
    ctx.add_check("s23", "floored list prices verify at 1e-12", bool(_as_flag(frames["policy"]["verifies"]).all()))
    ctx.add_check("s23", "no floor-study release is unverified", not _as_flag(gate_frame["unsafe_release"]).any())
    ctx.add_check("s23", "every leftover saving obeys the corollary bound", bool(summary["bound_holds"].all()))
    return frames


# ---------------------------------------------------------------------------
# Stage s24: exact projection reference and the non-ADMM solver
# ---------------------------------------------------------------------------


def _residuals(snapshot: core.ConstraintSnapshot, vector: np.ndarray) -> np.ndarray:
    """Row residuals A p - b, computed like the independent verifier (CSR segment sums)."""

    matrix = snapshot.matrix
    products = matrix.data * np.asarray(vector, dtype=np.float64)[matrix.indices]
    return np.add.reduceat(products, matrix.indptr[:-1]) - snapshot.bounds


def _dual_value(snapshot: core.ConstraintSnapshot, origin: np.ndarray, multipliers: np.ndarray) -> float:
    """Lagrange dual of min 1/2 ||p - p0||^2 s.t. A p <= b at lambda >= 0 (a lower bound on the optimum)."""

    lam = np.maximum(np.asarray(multipliers, dtype=np.float64), 0.0)
    lifted = np.asarray(snapshot.matrix.T @ lam).ravel()
    return float(lam @ _residuals(snapshot, origin) - 0.5 * float(lifted @ lifted))


def _active_set_multipliers(snapshot: core.ConstraintSnapshot, origin: np.ndarray, vector: np.ndarray, *,
                            slack: float = 1e-9) -> np.ndarray:
    """Multipliers on the near-active rows of a solution, from the stationarity condition.

    At the projection p*, p0 - p* = A^T lambda* with lambda* >= 0 supported on
    active rows. Least squares on the rows of ``vector`` with residual >= -slack,
    clipped at zero, approximates such multipliers when ``vector`` is close to
    p*. Any nonnegative vector yields a valid dual bound, so the bound stays
    sound when the approximation is loose; it is the fallback for the solvers'
    own dual vectors.
    """

    from scipy.sparse.linalg import lsmr

    residual = _residuals(snapshot, vector)
    active = np.flatnonzero(residual >= -slack)
    multipliers = np.zeros(int(snapshot.constraints), dtype=np.float64)
    if active.size == 0:
        return multipliers
    system = snapshot.matrix[active].T.tocsr()
    target = np.asarray(origin, dtype=np.float64) - np.asarray(vector, dtype=np.float64)
    solution = lsmr(system, target, atol=1e-14, btol=1e-14, maxiter=20000)[0]
    multipliers[active] = np.maximum(np.asarray(solution, dtype=np.float64), 0.0)
    return multipliers


def _refine_multipliers(snapshot: core.ConstraintSnapshot, origin: np.ndarray, multipliers: np.ndarray, *,
                        max_seconds: float = 3.0, max_sweeps: int = 5000) -> np.ndarray:
    """Hildreth's dual coordinate ascent, warm-started, with a time cap.

    Each coordinate step raises the dual value, so the result is never worse
    than the starting multipliers, and it stays nonnegative.
    """

    matrix = snapshot.matrix
    bounds = snapshot.bounds
    lam = np.maximum(np.asarray(multipliers, dtype=np.float64), 0.0).copy()
    point = np.asarray(origin, dtype=np.float64) - np.asarray(matrix.T @ lam).ravel()
    norms = np.add.reduceat(matrix.data * matrix.data, matrix.indptr[:-1])
    indptr, indices, data = matrix.indptr, matrix.indices, matrix.data
    started = time.perf_counter()
    for _ in range(int(max_sweeps)):
        residual = _residuals(snapshot, point)
        largest = 0.0
        for row in np.flatnonzero((residual > 0.0) | (lam > 0.0)):
            start, end = indptr[row], indptr[row + 1]
            columns, coefficients = indices[start:end], data[start:end]
            updated = max(0.0, lam[row] + (float(coefficients @ point[columns]) - bounds[row]) / norms[row])
            step = updated - lam[row]
            if step != 0.0:
                point[columns] -= step * coefficients
                lam[row] = updated
                largest = max(largest, abs(step))
        if largest < 1e-15 or time.perf_counter() - started > max_seconds:
            break
    return lam


def _strictly_feasible_blend(snapshot: core.ConstraintSnapshot, vector: np.ndarray, interior: np.ndarray) -> np.ndarray | None:
    """The point on the segment from ``vector`` toward a strictly feasible ``interior`` closest to ``vector``
    whose computed residuals are all <= 0, or None when ``interior`` is not strictly feasible."""

    interior_residual = _residuals(snapshot, interior)
    if interior_residual.max(initial=-math.inf) > 0.0:
        return None
    residual = _residuals(snapshot, vector)
    violated = residual > 0.0
    if not violated.any():
        return np.asarray(vector, dtype=np.float64).copy()
    step = float(np.max(residual[violated] / (residual[violated] - interior_residual[violated])))
    for factor in (1.0, 1.5, 2.0, 4.0, 16.0):
        weight = min(1.0, step * factor)
        candidate = (1.0 - weight) * vector + weight * interior
        if _residuals(snapshot, candidate).max(initial=-math.inf) <= 0.0:
            return candidate
    return np.asarray(interior, dtype=np.float64).copy()


def projection_certificate(snapshot: core.ConstraintSnapshot, origin: np.ndarray,
                           solutions: Mapping[str, np.ndarray],
                           duals: Mapping[str, np.ndarray | None],
                           interiors: Mapping[str, np.ndarray] | None = None) -> tuple[dict[str, Any], np.ndarray | None]:
    """Bracket the distance from p0 to the exact projection p* without trusting any solver.

    Lower bound: for every lambda >= 0, the dual value g(lambda) satisfies
    g(lambda) <= 1/2 ||p* - p0||^2 (weak duality). Candidates are the solvers'
    own dual vectors and multipliers recovered from the active rows of the
    solution with the smallest objective among those within 1e-12 of feasible.
    Upper bound: ||p* - p0|| <= ||q - p0|| for every q with A q <= b (computed
    residuals <= 0); candidates are such solutions, the given interior points,
    and blends of every other solution toward each of them. The closest such q
    also satisfies ||q - p*||^2 <= 2 (1/2 ||q - p0||^2 - g(lambda)) by strong
    convexity.
    """

    origin = np.asarray(origin, dtype=np.float64)

    def change(vector: np.ndarray) -> float:
        return float(np.linalg.norm(np.asarray(vector, dtype=np.float64) - origin))

    multipliers: dict[str, np.ndarray] = {}
    for name, lam in duals.items():
        if lam is not None and np.isfinite(lam).all():
            multipliers[f"dual:{name}"] = np.asarray(lam, dtype=np.float64)
    errors = []
    near_feasible = {name: v for name, v in solutions.items()
                     if _residuals(snapshot, v).max(initial=-math.inf) <= PRIMARY_TOLERANCE}
    pool = near_feasible or dict(solutions)
    if pool:
        best_primal = min(pool, key=lambda name: change(pool[name]))
        try:
            multipliers[f"active_rows:{best_primal}"] = _active_set_multipliers(snapshot, origin, pool[best_primal])
        except Exception as exc:  # noqa: BLE001 - a solver dual may still give the bound
            errors.append(f"{best_primal}: {type(exc).__name__}: {exc}")
    values = {name: _dual_value(snapshot, origin, lam) for name, lam in multipliers.items()}
    best_dual = max(values, key=values.get) if values else ""
    dual_bound = values.get(best_dual, -math.inf)
    if pool and best_dual:
        # Refine only when the best dual leaves more than 1e-8 relative gap to the best near-feasible objective.
        objective = 0.5 * change(pool[best_primal]) ** 2
        if objective - dual_bound > 1e-8 * max(objective, 1e-300):
            try:
                refined = _refine_multipliers(snapshot, origin, multipliers[best_dual])
                refined_value = _dual_value(snapshot, origin, refined)
                if refined_value > dual_bound:
                    best_dual, dual_bound = f"refined:{best_dual}", refined_value
            except Exception as exc:  # noqa: BLE001 - the unrefined bound stays valid
                errors.append(f"refinement: {type(exc).__name__}: {exc}")
    lower = math.sqrt(2.0 * dual_bound) if dual_bound > 0 else 0.0

    feasible: dict[str, np.ndarray] = {}
    candidates = {**{name: np.asarray(v, dtype=np.float64) for name, v in solutions.items()},
                  **{f"interior:{name}": np.asarray(v, dtype=np.float64) for name, v in (interiors or {}).items()}}
    strict = {name: v for name, v in candidates.items() if _residuals(snapshot, v).max(initial=-math.inf) <= 0.0}
    feasible.update(strict)
    for name, vector in solutions.items():
        if name in strict:
            continue
        for interior_name, interior in strict.items():
            blended = _strictly_feasible_blend(snapshot, np.asarray(vector, dtype=np.float64), interior)
            if blended is not None:
                feasible[f"{name} toward {interior_name}"] = blended
    point = None
    info: dict[str, Any] = {"dual_source": best_dual, "dual_value": dual_bound, "projection_distance_lower": lower,
                            "certificate_errors": "; ".join(errors)}
    if feasible:
        point_name = min(feasible, key=lambda key: change(feasible[key]))
        point = feasible[point_name]
        upper = change(point)
        gap = 0.5 * upper * upper - dual_bound
        info.update({"feasible_source": point_name, "projection_distance_upper": upper,
                     "certified_gap": gap,
                     "feasible_point_distance_bound": math.sqrt(2.0 * max(gap, 0.0)) if math.isfinite(gap) else math.nan,
                     "relative_width": upper / lower - 1.0 if lower > 0 else math.inf})
    else:
        info.update({"feasible_source": "", "projection_distance_upper": math.nan, "certified_gap": math.nan,
                     "feasible_point_distance_bound": math.nan, "relative_width": math.nan})
    return info, point


def s24_dataset(ctx, state) -> dict[str, pd.DataFrame]:
    cfg = ctx.config
    methods = list(cfg["projection_methods"])
    rows, distance_rows, certificate_rows = [], [], []
    proposals = state.ordered_proposals()[: int(cfg["projection_proposals"])]
    for unit, ((model, seed), proposal) in enumerate(proposals):
        for warmup in range(int(cfg["projection_warmups"])):
            for method in rev.rotate(methods, unit + warmup):
                run_method_ext(ctx, state.snapshot, proposal, method, tolerance=PRIMARY_TOLERANCE)
        vectors: dict[str, np.ndarray] = {}
        dual_vectors: dict[str, np.ndarray | None] = {}
        outcomes: dict[str, dict[str, Any]] = {}
        for repetition in range(int(cfg["projection_repetitions"])):
            ctx.beat(f"s24 {state.key}")
            for order_index, method in enumerate(rev.rotate(methods, unit + repetition)):
                outcome, prices, duals = split_outcome(
                    run_method_ext(ctx, state.snapshot, proposal, method, tolerance=PRIMARY_TOLERANCE))
                if repetition == 0:
                    vectors[method], dual_vectors[method], outcomes[method] = prices, duals, outcome
                rows.append({"dataset": state.name, "model": model, "seed": int(seed), "unit": unit,
                             "repetition": repetition, "execution_order": order_index, **outcome})
        started = time.perf_counter()
        usable = {m: v for m, v in vectors.items() if not outcomes[m].get("error")}
        # List prices satisfy every derivation row strictly and every decomposition
        # row with equality; adding a positive constant makes all rows strict.
        interior = state.list_prices + float(np.median(state.list_prices))
        info, point = projection_certificate(state.snapshot, proposal, usable, dual_vectors,
                                             {"list_prices_plus_median": interior})
        certificate_rows.append({"dataset": state.name, "model": model, "seed": int(seed), "unit": unit,
                                 "certificate_seconds": time.perf_counter() - started, **info})
        lower, upper = info["projection_distance_lower"], info["projection_distance_upper"]
        bound = info["feasible_point_distance_bound"]
        osqp = vectors.get("osqp_qp") if "osqp_qp" in usable else None
        for method, prices in vectors.items():
            l2 = float(np.linalg.norm(prices - proposal))
            l1 = float(np.abs(prices - proposal).sum())
            to_point = float(np.linalg.norm(prices - point)) if point is not None else math.nan
            distance_rows.append({
                "dataset": state.name, "model": model, "seed": int(seed), "unit": unit, "method": method,
                "verified": bool(outcomes[method]["verified"]), "status": str(outcomes[method].get("status", "")),
                "max_residual_after": float(outcomes[method]["max_residual_after"]),
                "l1_change": l1, "l2_change": l2,
                "l2_ratio_to_projection_low": l2 / upper if upper and math.isfinite(upper) else math.nan,
                "l2_ratio_to_projection_high": l2 / lower if lower > 0 else math.nan,
                "distance_to_projection_low": max(to_point - bound, 0.0) if math.isfinite(bound) else math.nan,
                "distance_to_projection_high": to_point + bound if math.isfinite(bound) else math.nan,
                "l1_ratio_to_osqp": l1 / max(float(np.abs(osqp - proposal).sum()), 1e-300) if osqp is not None else math.nan,
                "l2_ratio_to_osqp": l2 / max(float(np.linalg.norm(osqp - proposal)), 1e-300) if osqp is not None else math.nan,
            })
    return {"raw": pd.DataFrame(rows), "distances": pd.DataFrame(distance_rows),
            "certificates": pd.DataFrame(certificate_rows)}


def run_stage_24(ctx) -> dict[str, pd.DataFrame]:
    parts = [(f"{state.key}/projection", lambda state=state: s24_dataset(ctx, state)) for state in ctx.datasets.values()]
    frames = rev.run_parts(ctx, "s24", parts)
    raw, distances, certificates = frames["raw"], frames["distances"], frames["certificates"]
    frames["method_summary"] = rev.summarize_method_trials(raw, ["dataset", "method"])
    frames["paired"] = rev.paired_speedups(raw, cell_columns=["dataset"], unit_column="unit", repetition_column="repetition",
                                           reference="arbiter_active",
                                           baselines=[m for m in ctx.config["projection_methods"] if m != "arbiter_active"],
                                           resamples=int(ctx.config["bootstrap_resamples"]))
    d = distances.copy()
    d["verified"] = _as_flag(d["verified"])
    frames["summary"] = d.groupby(["method"], as_index=False).agg(
        proposals=("l2_change", "size"), verified=("verified", "sum"),
        l2_ratio_low_median=("l2_ratio_to_projection_low", "median"),
        l2_ratio_high_median=("l2_ratio_to_projection_high", "median"),
        l2_ratio_low_min=("l2_ratio_to_projection_low", "min"),
        l2_ratio_high_max=("l2_ratio_to_projection_high", "max"),
        distance_to_projection_high_median=("distance_to_projection_high", "median"),
        l1_ratio_to_osqp_median=("l1_ratio_to_osqp", "median"),
        l1_ratio_to_osqp_min=("l1_ratio_to_osqp", "min"),
        l1_ratio_to_osqp_max=("l1_ratio_to_osqp", "max"),
        l2_ratio_to_osqp_median=("l2_ratio_to_osqp", "median"))
    frames["certificate_summary"] = pd.DataFrame([{
        "proposals": len(certificates),
        "relative_width_median": float(certificates["relative_width"].median()),
        "relative_width_max": float(certificates["relative_width"].max()),
        "feasible_point_distance_bound_max": float(certificates["feasible_point_distance_bound"].max()),
        "certificate_seconds_max": float(certificates["certificate_seconds"].max()),
        "dual_sources": ";".join(sorted(set(certificates["dual_source"].fillna("").astype(str)))),
        "feasible_sources": ";".join(sorted(set(certificates["feasible_source"].fillna("").astype(str)))),
    }])
    rev.write_outputs(ctx, "s24", frames)
    ctx.add_check("s24", "no projection-study release is unverified", not _as_flag(raw["unsafe_release"]).any())
    ctx.add_check("s24", "every proposal has a two-sided bracket on its projection distance",
                  bool(len(certificates) and (certificates["projection_distance_lower"] > 0).all()
                       and np.isfinite(certificates["projection_distance_upper"]).all()
                       and (certificates["projection_distance_lower"] <= certificates["projection_distance_upper"] * (1 + 1e-12)).all()))
    ctx.add_check("s24", "the bracket is within 1 percent for every proposal",
                  bool(len(certificates) and (certificates["relative_width"] <= 0.01).all()), severity="finding",
                  detail=f"max relative width {float(certificates['relative_width'].max()):.3g}")
    arbiter = raw[raw["method"] == "arbiter_active"]
    ctx.add_check("s24", "Arbiter verifies every proposal at 1e-12", bool(_as_flag(arbiter["verified"]).all()),
                  severity="finding")
    return frames


# ---------------------------------------------------------------------------
# Stage s25: scale-out
# ---------------------------------------------------------------------------


class GeneratedWorkload:
    def __init__(self, name: str, assets: pd.DataFrame, relations: pd.DataFrame, list_prices: np.ndarray,
                 snapshot: core.ConstraintSnapshot, info: dict[str, Any]) -> None:
        self.name = name
        self.assets = assets
        self.asset_ids = assets["asset_id"].astype(str).tolist()
        self.relations = relations
        self.list_prices = list_prices
        self.snapshot = snapshot
        self.info = info


def build_generated_workload(ctx, dataset: str, size: int) -> GeneratedWorkload:
    specs = ctx.step2["table_specs"][dataset]
    setup_started = time.perf_counter()
    started = time.perf_counter_ns()
    generated = wl.generate_deterministic_workload(dataset=dataset, table_specs=specs, max_assets=int(size))
    generate_ms = (time.perf_counter_ns() - started) / 1e6
    assets = generated[generated["expected_supported"].astype(bool)].reset_index(drop=True)
    started = time.perf_counter_ns()
    descriptors = {str(a): descriptor_from_sql_ext(s) for a, s in zip(assets["asset_id"], assets["sql"])}
    parse_ms = (time.perf_counter_ns() - started) / 1e6
    parse_matches = all(descriptors[str(a)] == wl.QueryDescriptor.from_json(str(j))
                        for a, j in zip(assets["asset_id"], assets["oracle_descriptor"]))
    started = time.perf_counter_ns()
    relations = wl.infer_relations_from_descriptors(assets, descriptors)
    infer_ms = (time.perf_counter_ns() - started) / 1e6
    info: dict[str, Any] = {"dataset": dataset, "size": int(size), "assets": len(assets), "generate_ms": generate_ms,
                            "parse_ms": parse_ms, "infer_ms": infer_ms, "relations": len(relations),
                            "parse_matches_construction": bool(parse_matches)}
    for kind in rev.RELATION_FAMILIES:
        info[f"relations_{kind}"] = int((relations["relation_type"] == kind).sum())
    state = ctx.datasets[dataset]
    audited = ctx.step2["audited_relations"]
    reproduces = int(size) == len(state.asset_ids)
    if reproduces:
        same_assets = (assets["asset_id"].astype(str).tolist() == list(state.asset_ids)
                       and assets["sql"].astype(str).tolist() == state.assets["sql"].astype(str).tolist())
        same_relations = set(relations["relation_id"].astype(str)) == set(
            audited[audited["dataset"] == dataset]["relation_id"].astype(str))
    oracle = wl.oracle_relations(assets)
    truth = {wl._relation_key(r) for r in oracle.to_dict(orient="records")}
    found = {wl._relation_key(r) for r in relations.to_dict(orient="records")}
    tp = len(truth & found)
    info.update({"compared_with_audited": reproduces,
                 "reproduces_audited_workload": bool(same_assets) if reproduces else "",
                 "reproduces_audited_relations": bool(same_relations) if reproduces else "",
                 "oracle_relations": len(truth),
                 "oracle_precision": tp / len(found) if found else 1.0,
                 "oracle_recall": tp / len(truth) if truth else 1.0})
    features = ex.add_structural_features(assets.assign(estimated_rows=0.0))
    list_prices = ex.raw_controlled_policy(features)
    build_ms = []
    snapshot = None
    for _ in range(3):
        started = time.perf_counter_ns()
        snapshot = core.build_snapshot(assets["asset_id"].astype(str).tolist(), relations, epoch=1)
        build_ms.append((time.perf_counter_ns() - started) / 1e6)
    check = core.independent_verify(snapshot, list_prices, tolerance=PRIMARY_TOLERANCE)
    info.update({"rows": snapshot.constraints, "nonzeros": snapshot.nonzeros, "build_p50_ms": float(np.median(build_ms)),
                 "policy_verifies": bool(check.passed), "policy_max_residual": float(check.max_residual),
                 "numeric_bytes": int(snapshot.matrix.data.nbytes + snapshot.matrix.indices.nbytes
                                      + snapshot.matrix.indptr.nbytes + snapshot.bounds.nbytes),
                 "same_snapshot_digest_as_audited": bool(snapshot.constraint_digest == state.snapshot.constraint_digest)
                 if reproduces else "",
                 "setup_s": time.perf_counter() - setup_started})
    return GeneratedWorkload(f"{dataset}|{size}", assets, relations, list_prices, snapshot, info)


def combine_workloads(parts: Sequence[GeneratedWorkload], label: str) -> GeneratedWorkload:
    assets = pd.concat([p.assets for p in parts], ignore_index=True)
    relations = pd.concat([p.relations for p in parts], ignore_index=True)
    prices = np.concatenate([p.list_prices for p in parts])
    started = time.perf_counter_ns()
    snapshot = core.build_snapshot(assets["asset_id"].astype(str).tolist(), relations, epoch=1)
    build_ms = (time.perf_counter_ns() - started) / 1e6
    check = core.independent_verify(snapshot, prices, tolerance=PRIMARY_TOLERANCE)
    info = {"dataset": "combined", "size": label, "assets": len(assets), "relations": len(relations),
            "rows": snapshot.constraints, "nonzeros": snapshot.nonzeros, "build_p50_ms": build_ms,
            "setup_s": build_ms / 1000.0,
            "policy_verifies": bool(check.passed), "policy_max_residual": float(check.max_residual),
            "parse_matches_construction": all(bool(p.info["parse_matches_construction"]) for p in parts),
            "compared_with_audited": False, "reproduces_audited_workload": "", "reproduces_audited_relations": "",
            "oracle_precision": min(float(p.info["oracle_precision"]) for p in parts),
            "oracle_recall": min(float(p.info["oracle_recall"]) for p in parts),
            "numeric_bytes": int(snapshot.matrix.data.nbytes + snapshot.matrix.indices.nbytes
                                 + snapshot.matrix.indptr.nbytes + snapshot.bounds.nbytes)}
    return GeneratedWorkload(f"combined|{label}", assets, relations, prices, snapshot, info)


def scale_candidates(workload: GeneratedWorkload, *, cases: int, budget: float) -> list[dict[str, Any]]:
    families = sorted(set(workload.relations["relation_type"].astype(str)))
    pool = []
    for family in families:
        center, label = ex.relation_family_attack_center(workload.snapshot, workload.list_prices, family,
                                                         tolerance=PRIMARY_TOLERANCE)
        attacks = core.exact_box_row_attacks(workload.snapshot, center, budget_fraction=float(budget), limit=int(cases),
                                             tolerance=PRIMARY_TOLERANCE, include_types=[family])
        for attack in attacks:
            attack["center_label"] = label
        pool.extend(attacks)
    return pool


def scale_cell(ctx, workload: GeneratedWorkload, candidates: Sequence[dict[str, Any]]) -> pd.DataFrame:
    cfg = ctx.config
    methods = list(cfg["scale_out_methods"])
    rows = []
    for case_index, attack in enumerate(candidates):
        candidate = attack["candidate"]
        digest = hashlib.sha256(np.asarray(candidate, dtype="<f8").tobytes()).hexdigest()
        case_id = hashlib.sha256(f"{workload.name}|{attack['constraint_type']}|{attack['row_id']}".encode()).hexdigest()[:24]
        precheck = core.independent_verify(workload.snapshot, candidate, tolerance=PRIMARY_TOLERANCE)
        warmups = int(cfg["scale_out_fast_warmups"])
        for warmup in range(warmups):
            active = [m for m in methods if m in FAST_METHODS or warmup < int(cfg["scale_out_slow_warmups"])]
            for method in rev.rotate(active, case_index + warmup):
                run_method_ext(ctx, workload.snapshot, candidate, method, tolerance=PRIMARY_TOLERANCE)
        for repetition in range(int(cfg["scale_out_fast_repetitions"])):
            ctx.beat(f"s25 {workload.name}")
            active = [m for m in methods if m in FAST_METHODS or repetition < int(cfg["scale_out_slow_repetitions"])]
            for order_index, method in enumerate(rev.rotate(active, case_index + repetition)):
                outcome, _, _ = split_outcome(run_method_ext(ctx, workload.snapshot, candidate, method,
                                                             tolerance=PRIMARY_TOLERANCE))
                rows.append({"workload": workload.name, "dataset": workload.info["dataset"], "size": str(workload.info["size"]),
                             "assets": workload.snapshot.assets, "rows": workload.snapshot.constraints,
                             "attack_case_id": case_id, "case_index": case_index, "repetition": repetition,
                             "execution_order": order_index, "target_constraint_type": attack["constraint_type"],
                             "candidate_digest": digest, "violations_before": int(precheck.violations),
                             "constraint_digest": workload.snapshot.constraint_digest, **outcome})
    return pd.DataFrame(rows)


def run_stage_25(ctx) -> dict[str, pd.DataFrame]:
    cfg = ctx.config
    cache: dict[tuple[str, int], GeneratedWorkload] = {}

    def workload_for(dataset: str, size: int) -> GeneratedWorkload:
        key = (dataset, int(size))
        if key not in cache:
            cache[key] = build_generated_workload(ctx, dataset, int(size))
        return cache[key]

    def dataset_part(dataset: str, size: int) -> dict[str, pd.DataFrame]:
        workload = workload_for(dataset, size)
        candidates = scale_candidates(workload, cases=int(cfg["scale_out_cases_per_family"]),
                                      budget=float(cfg["scale_out_attack_budget"]))
        trials = scale_cell(ctx, workload, candidates)
        return {"workloads": pd.DataFrame([workload.info]), "raw": trials}

    def combined_part(size: int) -> dict[str, pd.DataFrame]:
        parts = [workload_for(d, size) for d in ctx.datasets]
        combined = combine_workloads(parts, f"{len(parts)}x{size}")
        offsets = np.cumsum([0] + [len(p.asset_ids) for p in parts])
        candidates = []
        for part, offset in zip(parts, offsets[:-1]):
            for attack in scale_candidates(part, cases=int(cfg["scale_out_combined_cases_per_family"]),
                                           budget=float(cfg["scale_out_attack_budget"])):
                vector = combined.list_prices.copy()
                vector[offset: offset + len(part.asset_ids)] = attack["candidate"]
                candidates.append({**attack, "candidate": vector, "row_id": f"{part.info['dataset']}|{attack['row_id']}"})
        trials = scale_cell(ctx, combined, candidates)
        return {"workloads": pd.DataFrame([combined.info]), "raw": trials}

    parts = []
    for dataset, sizes in cfg["scale_out_sizes"].items():
        if dataset not in ctx.datasets:
            continue
        for size in sizes:
            parts.append((f"{rev.slug(dataset)}/size-{int(size):05d}",
                          lambda dataset=dataset, size=size: dataset_part(dataset, int(size))))
    for size in cfg["scale_out_combined"]:
        parts.append((f"combined/size-{int(size):05d}", lambda size=size: combined_part(int(size))))
    frames = rev.run_parts(ctx, "s25", parts)
    raw = frames["raw"]
    frames["method_summary"] = rev.summarize_method_trials(raw, ["workload", "dataset", "size", "assets", "method"])
    frames["paired"] = rev.paired_speedups(raw, cell_columns=["workload"], unit_column="attack_case_id",
                                           repetition_column="repetition", reference="arbiter_active",
                                           baselines=[m for m in cfg["scale_out_methods"] if m != "arbiter_active"],
                                           resamples=int(cfg["bootstrap_resamples"]))
    frames["summary"] = frames["method_summary"][frames["method_summary"]["method"] == "arbiter_active"].reset_index(drop=True)
    rev.write_outputs(ctx, "s25", frames)
    workloads = frames["workloads"]
    ctx.add_check("s25", "every workload's list prices verify at 1e-12", bool(_as_flag(workloads["policy_verifies"]).all()))
    ctx.add_check("s25", "every generated asset parses to its construction descriptor",
                  bool(_as_flag(workloads["parse_matches_construction"]).all()))
    repro = workloads[_as_flag(workloads["compared_with_audited"])]
    ctx.add_check("s25", "the generator reproduces the audited 4,096-asset workloads, relations, and snapshots",
                  bool(len(repro) and _as_flag(repro["reproduces_audited_workload"]).all()
                       and _as_flag(repro["reproduces_audited_relations"]).all()
                       and _as_flag(repro["same_snapshot_digest_as_audited"]).all()),
                  severity="fatal" if len(repro) else "finding")
    ctx.add_check("s25", "compiler matches the oracle on every workload (precision and recall 1)",
                  bool((workloads["oracle_precision"] == 1.0).all() and (workloads["oracle_recall"] == 1.0).all()))
    ctx.add_check("s25", "no scale-out trial releases an unverified vector", not _as_flag(raw["unsafe_release"]).any())
    arbiter = raw[raw["method"] == "arbiter_active"]
    ctx.add_check("s25", "Arbiter verifies every scale-out candidate", bool(_as_flag(arbiter["verified"]).all()),
                  severity="finding")
    return frames


# ---------------------------------------------------------------------------
# Stage runner, estimate, and packaging
# ---------------------------------------------------------------------------

STAGE_FUNCTIONS: dict[str, Callable[[Any], dict[str, pd.DataFrame]]] = {
    "s21": run_stage_21,
    "s22": run_stage_22,
    "s23": run_stage_23,
    "s24": run_stage_24,
    "s25": run_stage_25,
}


def run_stage(ctx, stage: str) -> pd.DataFrame:
    ctx.log(f"Stage {stage} started: {STAGES[stage]}")
    started = time.perf_counter()
    outputs = STAGE_FUNCTIONS[stage](ctx)
    ctx.stage_wall_s[stage] = time.perf_counter() - started
    failures = [row for row in ctx.quality if row["stage"] == stage and row["severity"] == "fatal" and not row["passed"]]
    ctx.log(f"Stage {stage} finished in {ctx.stage_wall_s[stage] / 60:.1f} min")
    if failures:
        raise RuntimeError(f"Stage {stage} failed an integrity check: " + "; ".join(row["check"] for row in failures))
    return outputs.get(STAGE_HEADLINES[stage], pd.DataFrame())


def run_stage_guarded(ctx, stage: str) -> pd.DataFrame | None:
    try:
        return run_stage(ctx, stage)
    except Exception as exc:  # noqa: BLE001 - recorded, and the final cell refuses PASS
        text = traceback.format_exc()
        rev.atomic_write_text(ctx.output_dir / "errors" / f"{stage}.txt", text)
        ctx.add_check(stage, f"stage {stage} completed without an exception", False, detail=f"{type(exc).__name__}: {exc}")
        ctx.log(f"Stage {stage} FAILED: {type(exc).__name__}: {exc}. Later stages still run.")
        return None


def _scale_out_units(cfg: Mapping[str, Any]) -> float:
    """Relative cost of stage s25 (fallback when the pilot recorded no s25 outputs)."""

    families = len(rev.RELATION_FAMILIES)
    runs = int(cfg["scale_out_slow_repetitions"]) + int(cfg["scale_out_slow_warmups"])
    cases = families * int(cfg["scale_out_cases_per_family"]) * max(runs, 1)
    total = sum(cases * int(size) ** 1.25 for values in cfg["scale_out_sizes"].values() for size in values)
    combined = families * int(cfg["scale_out_combined_cases_per_family"]) * max(runs, 1)
    total += sum(3 * combined * (3 * int(size)) ** 1.25 for size in cfg["scale_out_combined"])
    return float(total)


def _scale_out_seconds(ctx) -> float | None:
    """Full-run seconds of stage s25 extrapolated from the pilot's measurements.

    Setup (generation, parsing, inference, oracle, snapshot builds) is scaled
    from the pilot's largest workload by size^1.5. Each method's median trial
    time is scaled by size^e, with e fitted between the pilot's two sizes and
    clipped to [1, 2], and multiplied by the full run's trial counts.
    """

    outputs = ctx.stage_outputs.get("s25")
    if not outputs or "raw" not in outputs or "workloads" not in outputs:
        return None
    raw, workloads = outputs["raw"], outputs["workloads"]
    per_dataset = workloads[workloads["dataset"].astype(str) != "combined"]
    if per_dataset.empty or "setup_s" not in per_dataset:
        return None
    reference = per_dataset.loc[per_dataset["assets"].astype(float).idxmax()]
    reference_size = float(reference["assets"])
    reference_setup = float(reference["setup_s"])
    medians = raw.groupby(["assets", "method"])["runtime_ms"].median()
    medians.index = pd.MultiIndex.from_tuples([(float(a), str(m)) for a, m in medians.index])
    sizes = sorted(float(size) for size in raw["assets"].unique())
    full = MODE_CONFIG["full"]
    families = len(rev.RELATION_FAMILIES)
    exponents, reference_ms = {}, {}
    for method in full["scale_out_methods"]:
        if (reference_size, method) not in medians.index:
            return None
        reference_ms[method] = float(medians[(reference_size, method)])
        exponents[method] = 1.25
        if len(sizes) >= 2 and (sizes[0], method) in medians.index and float(medians[(sizes[0], method)]) > 0:
            ratio = reference_ms[method] / float(medians[(sizes[0], method)])
            exponents[method] = min(max(math.log(max(ratio, 1e-9)) / math.log(reference_size / sizes[0]), 1.0), 2.0)
    plan = [(float(size), families * int(full["scale_out_cases_per_family"]), True)
            for values in full["scale_out_sizes"].values() for size in values]
    plan += [(3.0 * float(size), 3 * families * int(full["scale_out_combined_cases_per_family"]), False)
             for size in full["scale_out_combined"]]
    total = 0.0
    for size, cases, needs_setup in plan:
        scale = size / reference_size
        total += reference_setup * scale ** 1.5 if needs_setup else 0.1 * reference_setup * scale
        for method in full["scale_out_methods"]:
            if method in FAST_METHODS:
                runs = int(full["scale_out_fast_repetitions"]) + int(full["scale_out_fast_warmups"])
            else:
                runs = int(full["scale_out_slow_repetitions"]) + int(full["scale_out_slow_warmups"])
            total += cases * runs * reference_ms[method] / 1000.0 * scale ** exponents[method]
    return total


def estimate_full_runtime(ctx) -> pd.DataFrame:
    """Planning estimate of the full run from the pilot's part times (not a measurement).

    Each stage's pilot time is scaled by the factor of its dominant cost:
    model retraining (seeds) for s21 and s23, attacks for s22, and proposals
    times timed runs for s24. Stage s25 is extrapolated from its measured setup
    and trial times (see _scale_out_seconds). A 25 percent margin is added.
    """

    pilot, full = MODE_CONFIG["pilot"], MODE_CONFIG["full"]
    provenance = ctx.checkpointer.provenance()
    seconds = {}
    if len(provenance):
        valid = provenance[provenance["valid_for_this_run"].astype(bool)]
        seconds = valid.groupby("stage")["elapsed_s"].sum().to_dict()
    seeds = len(full["seeds"]) / len(pilot["seeds"])
    factors = {
        "s21": seeds * 1.2,
        "s22": (full["multipart_targets_per_dataset"] * len(full["multipart_attack_budgets"]))
        / (pilot["multipart_targets_per_dataset"] * len(pilot["multipart_attack_budgets"])) * 0.5,
        "s23": seeds,
        "s24": (full["projection_proposals"] * (full["projection_repetitions"] + full["projection_warmups"]))
        / (pilot["projection_proposals"] * max(pilot["projection_repetitions"] + pilot["projection_warmups"], 1)),
        "s25": _scale_out_units(full) / _scale_out_units(pilot),
    }
    measured_s25 = _scale_out_seconds(ctx)
    rows = []
    for stage in STAGE_ORDER:
        pilot_seconds = float(seconds.get(stage, ctx.stage_wall_s.get(stage, math.nan)))
        estimate = 1.25 * pilot_seconds * factors[stage]
        if stage == "s25" and measured_s25 is not None:
            estimate = 1.25 * measured_s25
            factors[stage] = estimate / max(pilot_seconds, 1e-9)
        rows.append({"stage": stage, "description": STAGES[stage], "pilot_seconds": pilot_seconds,
                     "factor": factors[stage], "estimated_full_seconds": estimate,
                     "estimated_full_minutes": estimate / 60.0})
    return pd.DataFrame(rows)


def run_folder(project_root: Path, mode: str, environment_label: str) -> Path:
    return Path(project_root) / "runs" / "step2_v1" / f"08_step2_{mode}_{rev.slug(environment_label)}"


def require_passed_pilot(project_root: Path, environment_label: str) -> dict[str, Any]:
    """The full run starts only after a passed pilot in the same environment."""

    folder = run_folder(project_root, "pilot", environment_label)
    pilot_zip = folder.parent / f"arbiter_step2_pilot_{rev.slug(environment_label)}_{SUITE_TAG}.zip"
    quality_path = folder / "stage_quality_checks.csv"
    if not (pilot_zip.is_file() and quality_path.is_file()):
        raise RuntimeError(
            "Run the pilot first: set RUN_MODE = \"pilot\", Run all, and wait for PILOT PASS. "
            f"Expected the pilot ZIP at {pilot_zip}."
        )
    quality = rev.read_csv(quality_path)
    failed = quality[(quality["severity"] == "fatal") & (~_as_flag(quality["passed"]))]
    if len(failed):
        raise RuntimeError("The pilot did not pass; do not start the full run. Failed: "
                           + "; ".join(failed["check"].astype(str).tolist()))
    hours = None  # the configuration digest rejects NaN and infinity
    estimate_path = folder / "runtime_estimate.csv"
    if estimate_path.is_file():
        estimate = rev.read_csv(estimate_path)
        value = float(estimate["estimated_full_seconds"].sum()) / 3600.0
        hours = round(value, 3) if math.isfinite(value) else None
    return {"pilot_zip": pilot_zip.name, "pilot_estimated_full_hours": hours}


def stage_input_files(project_root: Path, local_root: Path) -> dict[str, Any]:
    """Copy the public inputs from Drive to local disk; missing ones are downloaded later.

    DuckDB scans the NYC Parquet file in every query, so a local copy avoids
    repeated reads through the Drive mount. prepare_duckdb checks both files
    against their audited SHA-256 values before use.
    """

    import shutil

    local_root = Path(local_root)
    inputs = local_root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Any] = {"local_root": local_root, "nyc_parquet": None, "adult_zip": None}
    for key, name in [("nyc_parquet", "yellow_tripdata_2024-01.parquet"), ("adult_zip", "adult.zip")]:
        source = Path(project_root) / "inputs" / name
        target = inputs / name
        if source.is_file() and source.stat().st_size > 0:
            if not target.is_file() or target.stat().st_size != source.stat().st_size:
                shutil.copy2(source, target)
            paths[key] = target
    return paths


def headline_numbers(ctx) -> dict[str, Any]:
    headlines: dict[str, Any] = {}
    for stage, outputs in ctx.stage_outputs.items():
        summary = outputs.get("summary")
        if isinstance(summary, pd.DataFrame) and len(summary):
            headlines[stage] = json.loads(summary.to_json(orient="records"))
    return headlines


def finalize_step2(ctx, *, embedded_sources: Mapping[str, str], embedded_hashes: Mapping[str, str],
                   notebook_name: str, environment_label: str) -> dict[str, Any]:
    output_dir = ctx.output_dir
    unchanged = rev.snapshot_folder_state(ctx.package_dir) == ctx.package_state_before
    ctx.add_check("final", "audited-v4 package is byte-identical after the campaign", unchanged)
    missing = [stage for stage in STAGE_ORDER if stage not in ctx.stage_outputs]
    ctx.add_check("final", "every stage produced outputs", not missing, detail=", ".join(missing))
    quality = pd.DataFrame(ctx.quality)
    rev.atomic_write_csv(output_dir / "stage_quality_checks.csv", quality)
    rev.atomic_write_csv(output_dir / "input_audit.csv", ctx.input_audit)
    provenance = ctx.step2.get("database_provenance")
    if isinstance(provenance, pd.DataFrame):
        # File locations are local to the Colab runtime; the hashes identify the inputs.
        rev.atomic_write_csv(output_dir / "database_provenance.csv",
                             provenance.drop(columns=["local_file"], errors="ignore"))
    rev.atomic_write_csv(output_dir / "part_provenance.csv", ctx.checkpointer.provenance())
    if ctx.mode == "pilot":
        try:
            estimate = estimate_full_runtime(ctx)
        except Exception as exc:  # noqa: BLE001
            estimate = pd.DataFrame([{"stage": "all", "description": f"estimate failed: {exc}",
                                      "pilot_seconds": math.nan, "factor": math.nan,
                                      "estimated_full_seconds": math.inf, "estimated_full_minutes": math.inf}])
        rev.atomic_write_csv(output_dir / "runtime_estimate.csv", estimate)
    rev.atomic_write_text(output_dir / "headline_numbers.json",
                          json.dumps(headline_numbers(ctx), indent=2, sort_keys=True, default=str))
    rev.atomic_write_text(output_dir / "step2_run_config.json", json.dumps(ctx.config, indent=2, sort_keys=True))
    finger = rev.runtime_fingerprint(ctx.session_id)
    for name in ("sqlglot", "duckdb", "clarabel", "scikit-learn", "osqp"):
        finger.setdefault("packages", {})[name] = rev._package_version(name)
    rev.atomic_write_text(output_dir / "environment.json", json.dumps({
        **finger,
        "audited_input": {"package_dir_name": ctx.package_dir.name, "manifest_sha256": rev.AUDITED_MANIFEST_SHA256},
        "database": ctx.step2.get("database_label", ""),
        "notebook": notebook_name,
        "environment_label": environment_label,
        "stage_wall_seconds": ctx.stage_wall_s,
    }, indent=2, sort_keys=True, default=str))
    code_dir = output_dir / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    for name, text in embedded_sources.items():
        rev.atomic_write_text(code_dir / name, text)
    rev.atomic_write_text(code_dir / "code_manifest.json", json.dumps(dict(embedded_hashes), indent=2, sort_keys=True))
    fatal_failed = quality[(quality["severity"] == "fatal") & (~quality["passed"].astype(bool))]
    status = "PASS" if fatal_failed.empty else "FAILED"
    rev.atomic_write_text(output_dir / "README.md", "\n".join([
        f"# ARBITER Step 2 campaign ({ctx.mode}, {environment_label})", "",
        f"Suite version: {STEP2_SUITE_VERSION}. Status: {status}.", "",
        "Stages: GROUP BY extension (s21), multi-part audit (s22), price floors (s23),",
        "exact projection reference and Clarabel (s24), scale-out (s25).",
        "The audited-v4 package was read-only input and is byte-identical after the run",
        f"({'confirmed' if unchanged else 'NOT confirmed'}). Raw tables under outputs/ are the",
        "numerical authority. Prices and attacks are controlled constructions, not market data.", ""]))
    rows = []
    for path in sorted(output_dir.rglob("*")):
        relative = path.relative_to(output_dir)
        if not path.is_file() or relative.parts[0] == "parts" or path.name in {"checksums.csv", "checksums.sha256", "run.lock"}:
            continue
        if path.name.startswith("."):
            continue
        rows.append({"path": relative.as_posix(), "bytes": path.stat().st_size, "sha256": rev.sha256_file(path)})
    manifest = pd.DataFrame(rows)
    rev.atomic_write_csv(output_dir / "checksums.csv", manifest)
    rev.atomic_write_text(output_dir / "checksums.sha256",
                          "".join(f"{row.sha256}  {row.path}\n" for row in manifest.itertuples(index=False)))
    label = rev.slug(environment_label)
    suffix = "" if status == "PASS" else "_FAILED"
    zip_path = output_dir.parent / f"arbiter_step2_{ctx.mode}_{label}_{SUITE_TAG}{suffix}.zip"
    for stale in output_dir.parent.glob(f"arbiter_step2_{ctx.mode}_{label}_{SUITE_TAG}*.zip"):
        stale.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for row in manifest.itertuples(index=False):
            archive.write(output_dir / row.path, f"{output_dir.name}/{row.path}")
        for name in ["checksums.csv", "checksums.sha256"]:
            archive.write(output_dir / name, f"{output_dir.name}/{name}")
    with zipfile.ZipFile(zip_path) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Step 2 ZIP failed its integrity test")
        for row in manifest.itertuples(index=False):
            if rev.sha256_bytes(archive.read(f"{output_dir.name}/{row.path}")) != row.sha256:
                raise RuntimeError(f"Step 2 ZIP checksum mismatch: {row.path}")
    return {"status": status, "zip_path": zip_path, "zip_bytes": zip_path.stat().st_size,
            "fatal_failures": fatal_failed["check"].tolist(), "quality": quality}
