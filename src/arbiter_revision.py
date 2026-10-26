"""ARBITER revision campaign (Notebook 07), version 1.0.

Why this campaign exists
------------------------
The audited-v4 campaign (Notebooks 00-04), the request-bound closure
(Notebook 05 v1.1), and the performance hardening (Notebook 06 v1.0) verified
released price vectors with an absolute tolerance of 1e-7 in controlled price
units. Controlled list prices reach 1.25e-9, so a residual below 1e-7 can still
be a large share of a small list price. This module re-executes every
experiment whose outcome can depend on the verifier tolerance at 1e-12, keeps
1e-7 and 1e-9 as measured comparisons where they are informative, and adds four
replacement experiments (declaration coverage, pricing shapes, recompute versus
verify, and buyer history).

Integrity rules
---------------
* The audited-v4 evidence package is read-only input. Its manifest, its run
  configuration, every listed file, and the audited source files are verified
  by SHA-256 before any experiment starts.
* The audited modules (arbiter_core, arbiter_workload, arbiter_experiments) and
  the closure modules are imported unchanged. This module adds orchestration
  and measurement only.
* Measurement outcomes (rejection, fallback, solver failure, infeasible policy)
  are recorded, never hidden. Only integrity failures stop a run.
* Every CSV is read with float_precision="round_trip", so vectors read from the
  audited package equal the vectors the audited notebooks held in memory.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import datetime as _dt
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import secrets
import shutil
import socket
import sys
import tempfile
import time
from typing import Any, Callable, Iterable, Mapping, Sequence
import zipfile

import numpy as np
import pandas as pd

import arbiter_core as core
from arbiter_experiments import (
    add_structural_features,
    balanced_relation_closed_asset_ids,
    grouped_oof_predictions,
    raw_controlled_policy,
    relation_family_attack_center,
    relation_subset,
    run_certified_attack_campaign,
    run_fault_injection_campaign,
    run_omitted_relation_gap_campaign,
    run_stale_epoch_concurrency_campaign,
)
import arbiter_closure_experiments as closure


# ---------------------------------------------------------------------------
# Identity of the campaign and of its read-only inputs
# ---------------------------------------------------------------------------

REVISION_SUITE_VERSION = "2026-10-22-revision-v1.0"
AUDITED_SUITE_VERSION = "2026-08-01-audited-v4"
AUDITED_PACKAGE_NAME = "arbiter_icde_evidence_full_audited_v4"
AUDITED_MANIFEST_SHA256 = (
    "72a3843ca0c99077a80b4971646d765b3d9373178591bab1a451814f88a2ba80"
)
AUDITED_RUN_CONFIG_SHA256 = (
    "0c47860511da377b438d60da5117b4b9ad5d3f183383b247d14ffb4c3f8d51f8"
)
AUDITED_MANIFEST_FILES = 63
AUDITED_CODE_HASHES = dict(closure.AUDITED_CODE_HASHES)

AUDITED_TOLERANCE = 1e-7
PRIMARY_TOLERANCE = 1e-12

RELATION_FAMILIES = (
    "decomposition",
    "projection_derivation",
    "filter_derivation",
)
FAMILY_CODES = {
    "decomposition": "D",
    "projection_derivation": "P",
    "filter_derivation": "F",
}
MODEL_NAMES = ("dummy_median", "hist_gradient_boosting", "mlp")
SCALE_METHODS = (
    "reject_only",
    "arbiter_active",
    "full_cyclic",
    "osqp_qp",
    "osqp_qp_v4_stop",
)
BASELINE_METHODS = ("full_cyclic", "osqp_qp", "osqp_qp_v4_stop")
SHAPES = ("linear", "concave_sqrt", "tiered_8")

STAGES = {
    "s01": "natural proposals: tolerance sweep, leftover savings, heavy-load baselines",
    "s02": "exact per-row box attacks at 1e-12",
    "s03": "omitted-relation gap at 1e-12",
    "s04": "scale benchmark with paired baselines at 1e-12",
    "s05": "relation-family and active-row-budget ablation at 1e-12",
    "s06": "dense active-row budgets at 1e-7 and 1e-12",
    "s07": "fallback kernels at 1e-12",
    "s08": "declaration coverage (replaces the declaration-set table)",
    "s09": "seller pricing shapes (replaces the pricing-shape table)",
    "s10": "recompute versus verify (replaces the recompute table)",
    "s11": "buyer history (replaces the buyer-history row)",
    "s12": "protocol vectors and fault campaigns at 1e-12",
}
STAGE_VERSIONS = {stage: "1.0" for stage in STAGES}

STAGE_ORDER = tuple(STAGES)

MODE_CONFIG: dict[str, dict[str, Any]] = {
    "pilot": {
        "seeds": [11],
        "repair_sweeps": 400,
        "sweeps_to_verify_cap": 3000,
        "fallback_condition_sweeps": 80,
        "natural_tolerances": [1e-7, 1e-9, 1e-12],
        "heavy_methods": list(SCALE_METHODS),
        "heavy_repetitions": 1,
        "heavy_warmups": 0,
        "attack_budgets": [0.25],
        "attack_cases": 4,
        "attack_min_cases": 4,
        "gap_cases": 3,
        "scale_points": [128, 4096],
        "scale_cases_per_family": 1,
        "scale_repetitions": 2,
        "scale_warmups": 1,
        "scale_methods": list(SCALE_METHODS),
        "scale_attack_budget": 0.25,
        "osqp_qp_repetitions": 2,
        "osqp_qp_warmups": 1,
        "ablation_scale": 2048,
        "ablation_repetitions": 2,
        "ablation_warmups": 1,
        "dense_settings": [[1e-12, 80], [1e-12, 400], [1e-7, 80]],
        "dense_targets": [8, 192],
        "dense_budgets": [1, 8, 64],
        "dense_repetitions": 1,
        "dense_warmups": 0,
        "dense_budget_fraction": 0.50,
        "fallback_repetitions": 3,
        "fallback_warmups": 1,
        "fallback_sweeps": 20,
        "fallback_sweeps_grid": [20, 80],
        "coverage_family_subsets": ["none", "D", "DPF"],
        "coverage_fractions": [0.5],
        "coverage_fraction_seeds": [0],
        "shape_names": list(SHAPES),
        "shape_seeds": [11],
        "shape_models": list(MODEL_NAMES),
        "shape_attack_budget": 0.25,
        "shape_attack_cases": 4,
        "shape_tiers": 8,
        "recompute_repetitions": 20,
        "recompute_warmups": 2,
        "history_sizes": [16, 64],
        "history_buyers": 2,
        "history_variants": ["residual", "list_cap"],
        "protocol_concurrency_trials": 5,
        "closure_concurrency_trials": 3,
        "bootstrap_resamples": 2000,
        "runtime_go_limit_hours": 8.0,
    },
    "full": {
        "seeds": [11, 23, 37, 51, 73],
        "repair_sweeps": 400,
        "sweeps_to_verify_cap": 3000,
        "fallback_condition_sweeps": 80,
        "natural_tolerances": [1e-7, 1e-9, 1e-12],
        "heavy_methods": list(SCALE_METHODS),
        "heavy_repetitions": 3,
        "heavy_warmups": 1,
        "attack_budgets": [0.10, 0.25, 0.50],
        "attack_cases": 80,
        "attack_min_cases": 20,
        "gap_cases": 20,
        "scale_points": [128, 512, 1024, 2048, 4096],
        "scale_cases_per_family": 4,
        "scale_repetitions": 10,
        "scale_warmups": 3,
        "scale_methods": list(SCALE_METHODS),
        "scale_attack_budget": 0.25,
        "osqp_qp_repetitions": 10,
        "osqp_qp_warmups": 3,
        "ablation_scale": 2048,
        "ablation_repetitions": 10,
        "ablation_warmups": 3,
        "dense_settings": [[1e-12, 80], [1e-12, 400], [1e-7, 80]],
        "dense_targets": [8, 192],
        "dense_budgets": [1, 4, 8, 16, 32, 64],
        "dense_repetitions": 5,
        "dense_warmups": 2,
        "dense_budget_fraction": 0.50,
        "fallback_repetitions": 50,
        "fallback_warmups": 5,
        "fallback_sweeps": 20,
        "fallback_sweeps_grid": [20, 80],
        "coverage_family_subsets": [
            "none", "D", "P", "F", "DP", "DF", "PF", "DPF",
        ],
        "coverage_fractions": [0.25, 0.5, 0.75],
        "coverage_fraction_seeds": [0, 1, 2],
        "shape_names": list(SHAPES),
        "shape_seeds": [11, 23, 37, 51, 73],
        "shape_models": list(MODEL_NAMES),
        "shape_attack_budget": 0.25,
        "shape_attack_cases": 20,
        "shape_tiers": 8,
        "recompute_repetitions": 200,
        "recompute_warmups": 20,
        "history_sizes": [16, 64, 256],
        "history_buyers": 10,
        "history_variants": ["residual", "list_cap"],
        "protocol_concurrency_trials": 100,
        "closure_concurrency_trials": 50,
        "bootstrap_resamples": 10000,
        "runtime_go_limit_hours": 8.0,
    },
}


# Preregistered rule for the full run, fixed before any full-mode data exist:
# the mean OSQP (eps_abs 1e-12) cost per call that the pilot measured at its
# largest scale sets how many OSQP timing repetitions and warm-ups stage s04
# uses. Every other method keeps 10 repetitions and 3 warm-ups.
OSQP_REPETITION_RULE = (
    (1500.0, 10, 3),
    (6000.0, 5, 2),
    (math.inf, 3, 1),
)


def pilot_folder(project_root: Path, environment_label: str) -> Path:
    return Path(project_root) / "runs" / "revision_v1" / f"07_revision_pilot_{slug(environment_label)}"


def full_mode_osqp_rule(project_root: Path, environment_label: str) -> dict[str, Any]:
    """Apply OSQP_REPETITION_RULE to the passed pilot of this environment."""

    folder = pilot_folder(project_root, environment_label)
    pilot_zip = folder.parent / f"arbiter_revision_pilot_{slug(environment_label)}_v1_0.zip"
    scale_path = folder / "outputs" / "s04_scale_raw.csv"
    quality_path = folder / "stage_quality_checks.csv"
    if not (pilot_zip.is_file() and scale_path.is_file() and quality_path.is_file()):
        raise RuntimeError(
            "Run the pilot first: set RUN_MODE = \"pilot\", Run all, and wait for "
            "PILOT PASS. The full run reads the pilot's OSQP timing to fix its "
            f"OSQP repetitions. Expected the pilot ZIP at {pilot_zip}."
        )
    quality = read_csv(quality_path)
    failed = quality[(quality["severity"] == "fatal") & (~as_bool(quality["passed"]))]
    if len(failed):
        raise RuntimeError("The pilot did not pass; do not start the full run. Failed: "
                           + "; ".join(failed["check"].astype(str).tolist()))
    raw = read_csv(scale_path)
    osqp = raw[raw["method"] == "osqp_qp"]
    if osqp.empty:
        raise RuntimeError("the pilot has no OSQP scale trials")
    largest = int(osqp["scale"].max())
    mean_ms = float(osqp.loc[osqp["scale"] == largest, "runtime_ms"].mean())
    for limit, repetitions, warmups in OSQP_REPETITION_RULE:
        if mean_ms <= limit:
            break
    estimate_path = folder / "runtime_estimate.csv"
    estimated_hours = math.nan
    if estimate_path.is_file():
        estimate = read_csv(estimate_path)
        if "estimated_full_seconds" in estimate:
            estimated_hours = float(estimate["estimated_full_seconds"].sum()) / 3600.0
    return {
        "pilot_estimated_full_hours": estimated_hours,
        "pilot_osqp_mean_ms_at_largest_scale": mean_ms,
        "largest_pilot_scale": largest,
        "osqp_qp_repetitions": int(repetitions),
        "osqp_qp_warmups": int(warmups),
        "rule": [
            ["inf" if math.isinf(limit) else float(limit), int(reps), int(warm)]
            for limit, reps, warm in OSQP_REPETITION_RULE
        ],
    }


def mode_config(mode: str, override: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return the configuration of one run mode.

    ``override`` exists only for offline tests of this module; the notebook
    never passes it in Google Colab.
    """

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
    config["audited_tolerance"] = AUDITED_TOLERANCE
    config["revision_suite_version"] = REVISION_SUITE_VERSION
    return config


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_digest(value: Any) -> str:
    return sha256_bytes(core.canonical_json(value))


def slug(text: Any) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")
    return value[:80] or "item"


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV with exact float round-tripping."""

    try:
        return pd.read_csv(path, float_precision="round_trip", low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def as_bool(series: pd.Series) -> pd.Series:
    """Parse booleans that may have been round-tripped through CSV."""

    if series.dtype == bool:
        return series
    mapped = series.map(
        lambda value: value
        if isinstance(value, (bool, np.bool_))
        else {"true": True, "false": False}.get(str(value).strip().lower())
    )
    if mapped.isna().any():
        raise ValueError(f"column {series.name!r} contains non-boolean values")
    return mapped.astype(bool)


def rotate(sequence: Sequence[Any], offset: int) -> list[Any]:
    items = list(sequence)
    if not items:
        return items
    offset %= len(items)
    return items[offset:] + items[:offset]


def stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256("\x1f".join(str(part) for part in parts).encode("utf-8"))
    return int.from_bytes(digest.digest()[:8], "big") % (2**32)


def stable_rank_key(*parts: Any) -> str:
    return hashlib.sha256("\x1f".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def bootstrap_median_ci(
    values: Sequence[float], *, resamples: int, seed: int
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return (math.nan, math.nan)
    if array.size == 1:
        return (float(array[0]), float(array[0]))
    rng = np.random.default_rng(int(seed))
    medians = np.median(
        rng.choice(array, size=(int(resamples), array.size), replace=True), axis=1
    )
    low, high = np.quantile(medians, [0.025, 0.975])
    return float(low), float(high)


def q95(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    return float(np.quantile(array, 0.95)) if array.size else math.nan


def q99(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    return float(np.quantile(array, 0.99)) if array.size else math.nan


def p95_observed(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    return float(np.quantile(array, 0.95, method="higher")) if array.size else math.nan


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def runtime_fingerprint(session_id: str) -> dict[str, Any]:
    cpu_model = "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                cpu_model = line.split(":", 1)[1].strip()
                break
    memory_bytes = None
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("MemTotal:"):
                memory_bytes = int(line.split()[1]) * 1024
                break
    return {
        "session_id": session_id,
        "host_digest": sha256_bytes(socket.gethostname().encode("utf-8"))[:12],
        "cpu_model": cpu_model,
        "logical_cpu_count": os.cpu_count(),
        "memory_bytes": memory_bytes,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            name: _package_version(name)
            for name in [
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "osqp",
                "matplotlib",
            ]
        },
        "osqp_backend_loaded": bool(core.osqp is not None),
        "osqp_module": getattr(core.osqp, "__name__", "none"),
    }


# ---------------------------------------------------------------------------
# Run lock and checkpoints
# ---------------------------------------------------------------------------


class RunLock:
    """Stops a second Colab session from writing into the same output folder."""

    def __init__(
        self,
        path: Path,
        *,
        session_id: str,
        stale_seconds: int = 300,
        take_over: bool = False,
    ) -> None:
        self.path = Path(path)
        self.session_id = session_id
        self.stale_seconds = int(stale_seconds)
        self.take_over = bool(take_over)
        self._last_beat = 0.0
        self.started_utc = utc_now()

    def acquire(self) -> None:
        if self.path.is_file():
            try:
                info = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                info = {}
            other = str(info.get("session_id", ""))
            age = time.time() - float(info.get("heartbeat_unix", 0.0))
            if other and other != self.session_id and age < self.stale_seconds and not self.take_over:
                raise RuntimeError(
                    "Another Colab session wrote to this run folder "
                    f"{int(age)} seconds ago (session {other}). Running two "
                    "copies at once would corrupt the checkpoints. Close the "
                    "other tab and wait 5 minutes, or, if you are sure no "
                    "other copy is running, set TAKE_OVER_LOCK = True and run "
                    "all cells again."
                )
        self.beat("acquired", force=True)

    def beat(self, note: str = "", *, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_beat < 30.0:
            return
        self._last_beat = now
        atomic_write_text(
            self.path,
            json.dumps(
                {
                    "session_id": self.session_id,
                    "started_utc": self.started_utc,
                    "heartbeat_unix": now,
                    "heartbeat_utc": utc_now(),
                    "note": note,
                },
                indent=2,
            ),
        )

    def release(self) -> None:
        try:
            info = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return
        if info.get("session_id") == self.session_id:
            self.path.unlink(missing_ok=True)


class Checkpointer:
    """Stores each finished part of a stage as CSV files plus a marker."""

    MARKER = "_DONE.json"

    def __init__(
        self,
        parts_dir: Path,
        *,
        mode: str,
        input_digest: str,
        config_digest: str,
        session_id: str,
    ) -> None:
        self.parts_dir = Path(parts_dir)
        self.mode = mode
        self.input_digest = input_digest
        self.config_digest = config_digest
        self.session_id = session_id
        self.parts_dir.mkdir(parents=True, exist_ok=True)

    def part_dir(self, part: str) -> Path:
        return self.parts_dir.joinpath(*[slug(piece) for piece in part.split("/")])

    def key(self, part: str) -> dict[str, Any]:
        stage = part.split("/")[0]
        return {
            "part": part,
            "stage": stage,
            "stage_version": STAGE_VERSIONS[stage],
            "mode": self.mode,
            "input_digest": self.input_digest,
            "config_digest": self.config_digest,
            "revision_suite_version": REVISION_SUITE_VERSION,
        }

    def done(self, part: str) -> bool:
        directory = self.part_dir(part)
        marker = directory / self.MARKER
        if not marker.is_file():
            return False
        try:
            meta = json.loads(marker.read_text(encoding="utf-8"))
        except Exception:
            return False
        if meta.get("key") != self.key(part):
            return False
        for name, info in meta.get("files", {}).items():
            path = directory / name
            if not path.is_file() or sha256_file(path) != info.get("sha256"):
                return False
        return True

    def save(
        self,
        part: str,
        frames: Mapping[str, pd.DataFrame],
        *,
        elapsed_s: float,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        directory = self.part_dir(part)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / self.MARKER).unlink(missing_ok=True)
        files: dict[str, Any] = {}
        for name, frame in frames.items():
            if not isinstance(frame, pd.DataFrame):
                raise TypeError(f"part {part}: {name} is not a DataFrame")
            path = directory / f"{name}.csv"
            atomic_write_csv(path, frame)
            files[path.name] = {"sha256": sha256_file(path), "rows": int(len(frame))}
        meta = {
            "key": self.key(part),
            "files": files,
            "elapsed_s": float(elapsed_s),
            "finished_utc": utc_now(),
            "environment": runtime_fingerprint(self.session_id),
            "extra": dict(extra or {}),
        }
        atomic_write_text(
            directory / self.MARKER, json.dumps(meta, indent=2, sort_keys=True)
        )

    def load(self, part: str) -> dict[str, pd.DataFrame]:
        directory = self.part_dir(part)
        meta = json.loads((directory / self.MARKER).read_text(encoding="utf-8"))
        return {
            Path(name).stem: read_csv(directory / name)
            for name in sorted(meta.get("files", {}))
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
                    "stage_version": key.get("stage_version"),
                    "valid_for_this_run": key == self.key(str(key.get("part", "s01/x")))
                    if key.get("stage") in STAGE_VERSIONS
                    else False,
                    "elapsed_s": meta.get("elapsed_s"),
                    "finished_utc": meta.get("finished_utc"),
                    "session_id": environment.get("session_id"),
                    "host_digest": environment.get("host_digest"),
                    "cpu_model": environment.get("cpu_model"),
                    "logical_cpu_count": environment.get("logical_cpu_count"),
                    "python": environment.get("python"),
                    "numpy": environment.get("packages", {}).get("numpy"),
                    "pandas": environment.get("packages", {}).get("pandas"),
                    "scipy": environment.get("packages", {}).get("scipy"),
                    "scikit_learn": environment.get("packages", {}).get("scikit-learn"),
                    "osqp": environment.get("packages", {}).get("osqp"),
                }
            )
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Read-only audited input
# ---------------------------------------------------------------------------

_PACKAGE_MARKERS = (
    "run_config.json",
    "checksums.csv",
    "checksums.sha256",
    "04_evidence/all_stage_quality_checks.csv",
    "code/arbiter_core.py",
)


def is_audited_package(path: Path) -> bool:
    path = Path(path)
    return path.is_dir() and all((path / marker).is_file() for marker in _PACKAGE_MARKERS)


def _find_package_below(root: Path, max_depth: int = 4) -> Path | None:
    root = Path(root)
    if is_audited_package(root):
        return root
    frontier = [root]
    for _ in range(max_depth):
        next_frontier = []
        for directory in frontier:
            try:
                children = sorted(child for child in directory.iterdir() if child.is_dir())
            except OSError:
                continue
            for child in children:
                if child.name == "__pycache__":
                    continue
                if is_audited_package(child):
                    return child
                next_frontier.append(child)
        frontier = next_frontier
    return None


def safe_extract_zip(zip_path: Path, destination: Path) -> Path:
    """Extract an evidence ZIP after rejecting unsafe or oversized members."""

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise RuntimeError(f"{zip_path.name} contains duplicate members")
        if len(infos) > 20000:
            raise RuntimeError(f"{zip_path.name} has an unexpected member count")
        if sum(info.file_size for info in infos) > 4 * 1024**3:
            raise RuntimeError(f"{zip_path.name} exceeds the 4 GiB safety limit")
        for info in infos:
            member = Path(info.filename)
            mode = (info.external_attr >> 16) & 0o170000
            if (
                info.flag_bits & 0x1
                or info.filename.startswith(("/", "\\"))
                or "\\" in info.filename
                or member.is_absolute()
                or ".." in member.parts
                or mode == 0o120000
            ):
                raise RuntimeError(f"unsafe member in {zip_path.name}: {info.filename}")
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"corrupt member in {zip_path.name}: {bad}")
        archive.extractall(destination)
    found = _find_package_below(destination)
    if found is None:
        raise RuntimeError(
            f"{zip_path.name} does not contain the audited-v4 evidence package"
        )
    return found


def locate_audited_package(
    project_root: Path,
    manual_path: str | os.PathLike[str] | None,
    extract_root: Path,
) -> tuple[Path, str]:
    """Find the audited-v4 package folder or ZIP without modifying it."""

    project_root = Path(project_root)
    if manual_path:
        candidate = Path(manual_path)
        if candidate.is_dir():
            found = _find_package_below(candidate)
            if found is None:
                raise FileNotFoundError(
                    f"MANUAL_V4_PACKAGE folder has no audited package: {candidate}"
                )
            return found, "manual_folder"
        if candidate.is_file() and candidate.suffix.lower() == ".zip":
            return safe_extract_zip(candidate, extract_root), "manual_zip"
        raise FileNotFoundError(f"MANUAL_V4_PACKAGE does not exist: {candidate}")

    run_dir = project_root / "runs" / "full_audited_v4"
    nested = run_dir / AUDITED_PACKAGE_NAME
    if is_audited_package(nested):
        return nested, "drive_folder"
    notebook_zip = run_dir / f"{AUDITED_PACKAGE_NAME}.zip"
    if notebook_zip.is_file():
        return safe_extract_zip(notebook_zip, extract_root), "drive_notebook04_zip"
    zip_candidates: list[Path] = []
    for folder in [project_root / "inputs", project_root, run_dir]:
        if folder.is_dir():
            zip_candidates.extend(sorted(folder.glob(f"{AUDITED_PACKAGE_NAME}*.zip")))
    for candidate in zip_candidates:
        try:
            return safe_extract_zip(candidate, extract_root / slug(candidate.stem)), "drive_zip"
        except RuntimeError:
            continue
    raise FileNotFoundError(
        "Could not find the audited-v4 evidence package. Expected the folder "
        f"{nested} (created by Notebook 04), or its ZIP. If you only have the "
        "ZIP on your computer, upload it to MyDrive/Arbiter_ICDE_2027/inputs/ "
        "and run all cells again."
    )


def verify_audited_package(package_dir: Path) -> pd.DataFrame:
    """Verify identity and integrity of the read-only audited-v4 package."""

    package_dir = Path(package_dir).resolve()
    rows: list[dict[str, Any]] = []

    def add(scope: str, path: str, expected: str, observed: str, passed: bool) -> None:
        rows.append(
            {
                "scope": scope,
                "path": path,
                "expected_sha256": expected,
                "observed_sha256": observed,
                "passed": bool(passed),
            }
        )

    manifest_hash = sha256_file(package_dir / "checksums.sha256")
    add(
        "identity",
        "checksums.sha256",
        AUDITED_MANIFEST_SHA256,
        manifest_hash,
        manifest_hash == AUDITED_MANIFEST_SHA256,
    )
    config_hash = sha256_file(package_dir / "run_config.json")
    add(
        "identity",
        "run_config.json",
        AUDITED_RUN_CONFIG_SHA256,
        config_hash,
        config_hash == AUDITED_RUN_CONFIG_SHA256,
    )
    identity = pd.DataFrame(rows)
    if not identity["passed"].all():
        raise RuntimeError(
            "The audited-v4 package on Drive is not the audited version "
            "(manifest or run_config hash differs). Do not edit files inside it."
        )
    try:
        audit = closure.verify_audited_inputs(
            package_dir, package_dir / "code", active_run_dir=None
        )
    except (RuntimeError, FileNotFoundError) as exc:
        raise RuntimeError(
            f"The audited-v4 evidence at {package_dir} does not match its checksums ({exc}). "
            "Do not edit files inside that folder. To recover, upload your downloaded "
            "audited-v4 ZIP to MyDrive/Arbiter_ICDE_2027/inputs/, set MANUAL_V4_PACKAGE to "
            "its full path, and run all cells again."
        ) from exc
    audit = audit.copy()
    manifest = read_csv(package_dir / "checksums.csv")
    if len(manifest) != AUDITED_MANIFEST_FILES:
        raise RuntimeError(
            f"audited manifest lists {len(manifest)} files; expected {AUDITED_MANIFEST_FILES}"
        )
    listed = set(manifest["path"].astype(str))
    extras = []
    for path in sorted(package_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(package_dir).as_posix()
        if relative in listed or relative in {"checksums.csv", "checksums.sha256"}:
            continue
        extras.append(
            {
                "scope": "extra_file_not_in_manifest",
                "path": relative,
                "expected_sha256": "",
                "observed_sha256": sha256_file(path),
                "observed_bytes": path.stat().st_size,
                # Python bytecode written when a later notebook imported the
                # packaged modules is harmless and is not used here.
                "passed": "__pycache__" in Path(relative).parts,
            }
        )
    pieces = [identity, audit] + ([pd.DataFrame(extras)] if extras else [])
    combined = pd.concat(pieces, ignore_index=True, sort=False)
    unexpected = [
        row["path"]
        for row in extras
        if not row["passed"]
    ]
    if unexpected:
        raise RuntimeError(
            "Unexpected files were added inside the audited-v4 package: "
            + ", ".join(unexpected[:5])
        )
    return combined


def snapshot_folder_state(folder: Path) -> dict[str, str]:
    """Hash every file below a folder (used to prove the input is unchanged)."""

    folder = Path(folder)
    return {
        path.relative_to(folder).as_posix(): sha256_file(path)
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


# ---------------------------------------------------------------------------
# Relation evaluation and leftover savings
# ---------------------------------------------------------------------------


class RelationEvaluator:
    """Residuals and buyer savings over the encoded relation rows of a snapshot.

    For a decomposition row, a positive residual p_t - sum(p_S) is the amount a
    buyer saves by buying the components instead of the target. For a
    derivation row, a positive residual p_derived - p_source is the amount saved
    by buying the source. Savings are reported relative to the seller's list
    price of the target (the controlled policy price), and conservatively
    relative to max(posted price, list price). Residual divided by the posted
    price alone is ill-conditioned, because repair may move a posted price to
    zero within tolerance, so it is not used.
    """

    def __init__(self, snapshot: core.ConstraintSnapshot, list_prices: np.ndarray) -> None:
        self.snapshot = snapshot
        meta = snapshot.row_metadata
        types = meta["constraint_type"].astype(str).to_numpy()
        self.relation_rows = np.flatnonzero(np.isin(types, RELATION_FAMILIES))
        index = {asset_id: position for position, asset_id in enumerate(snapshot.asset_ids)}
        targets = meta["target_id"].astype(str).to_numpy()
        self.target_index = np.asarray(
            [index[targets[row]] for row in self.relation_rows], dtype=np.int64
        )
        self.row_ids = meta["row_id"].astype(str).to_numpy()[self.relation_rows]
        self.row_types = types[self.relation_rows]
        self.list_prices = np.asarray(list_prices, dtype=np.float64)
        if self.list_prices.shape != (snapshot.assets,):
            raise ValueError("list prices do not match the snapshot asset order")
        target_list = self.list_prices[self.target_index]
        if not np.isfinite(target_list).all() or (target_list <= 0.0).any():
            raise ValueError("relation targets need finite positive list prices")
        self.target_list = target_list

    def residuals(self, prices: np.ndarray) -> np.ndarray:
        matrix = self.snapshot.matrix
        vector = np.asarray(prices, dtype=np.float64)
        products = matrix.data * vector[matrix.indices]
        sums = np.add.reduceat(products, matrix.indptr[:-1])
        return sums - self.snapshot.bounds

    def metrics(self, prices: np.ndarray) -> dict[str, Any]:
        vector = np.asarray(prices, dtype=np.float64)
        residual = self.residuals(vector)[self.relation_rows]
        positive = np.maximum(residual, 0.0)
        saving = positive / self.target_list
        posted = vector[self.target_index]
        conservative = positive / np.maximum(np.maximum(posted, self.target_list), 1e-300)
        argmax = int(np.argmax(saving)) if saving.size else -1
        return {
            "relations_evaluated": int(residual.size),
            "positive_residual_relations": int(np.count_nonzero(residual > 0.0)),
            "relations_above_1e12": int(np.count_nonzero(residual > PRIMARY_TOLERANCE)),
            "max_relation_residual": float(residual.max()) if residual.size else math.nan,
            "max_saving_vs_list": float(saving.max()) if saving.size else math.nan,
            "n_saving_gt_1pct": int(np.count_nonzero(saving > 1e-2)),
            "n_saving_gt_0p1pct": int(np.count_nonzero(saving > 1e-3)),
            "max_saving_conservative": float(conservative.max()) if conservative.size else math.nan,
            "n_conservative_gt_1pct": int(np.count_nonzero(conservative > 1e-2)),
            "n_conservative_gt_0p1pct": int(np.count_nonzero(conservative > 1e-3)),
            "argmax_relation_id": str(self.row_ids[argmax]) if argmax >= 0 else "",
            "argmax_relation_type": str(self.row_types[argmax]) if argmax >= 0 else "",
        }

    def top_relations(self, prices: np.ndarray, k: int = 10) -> list[dict[str, Any]]:
        vector = np.asarray(prices, dtype=np.float64)
        residual = self.residuals(vector)[self.relation_rows]
        saving = np.maximum(residual, 0.0) / self.target_list
        order = np.argsort(-saving, kind="stable")[: int(k)]
        records = []
        for rank, position in enumerate(order):
            if saving[position] <= 0.0:
                break
            records.append(
                {
                    "rank": rank,
                    "relation_id": str(self.row_ids[position]),
                    "relation_type": str(self.row_types[position]),
                    "residual": float(residual[position]),
                    "target_list_price": float(self.target_list[position]),
                    "target_posted_price": float(vector[self.target_index[position]]),
                    "saving_vs_list": float(saving[position]),
                }
            )
        return records


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


@dataclass
class DatasetState:
    name: str
    key: str
    assets: pd.DataFrame
    asset_ids: list[str]
    relations: pd.DataFrame
    policy_frame: pd.DataFrame
    list_prices: np.ndarray
    features: pd.DataFrame
    snapshot: core.ConstraintSnapshot
    evaluator: RelationEvaluator
    proposals: dict[tuple[str, int], np.ndarray]

    def ordered_proposals(self) -> list[tuple[tuple[str, int], np.ndarray]]:
        return sorted(self.proposals.items(), key=lambda item: (item[0][0], item[0][1]))


@dataclass
class RevisionContext:
    mode: str
    config: dict[str, Any]
    audited_config: dict[str, Any]
    package_dir: Path
    output_dir: Path
    checkpointer: Checkpointer
    lock: RunLock | None
    session_id: str
    datasets: dict[str, DatasetState]
    assets_all: pd.DataFrame
    relations_all: pd.DataFrame
    policy_all: pd.DataFrame
    oof_all: pd.DataFrame
    pooled_metrics_all: pd.DataFrame
    audited_attacks: pd.DataFrame
    audited_gap: pd.DataFrame
    input_audit: pd.DataFrame
    package_state_before: dict[str, str]
    log_fn: Callable[[str], None] = print
    quality: list[dict[str, Any]] = field(default_factory=list)
    stage_wall_s: dict[str, float] = field(default_factory=dict)
    stage_outputs: dict[str, dict[str, pd.DataFrame]] = field(default_factory=dict)
    _bundles: dict[float, dict[str, Any]] = field(default_factory=dict)

    @property
    def outputs_dir(self) -> Path:
        return self.output_dir / "outputs"

    def log(self, message: str) -> None:
        self.log_fn(f"[{_dt.datetime.now().strftime('%H:%M:%S')}] {message}")

    def beat(self, note: str = "") -> None:
        if self.lock is not None:
            self.lock.beat(note)

    def gate_options(self) -> dict[str, Any]:
        return {
            # Revision sweep budget (audited: 80, sized for 1e-7). Cyclic
            # repair converges linearly, so 1e-12 needs more sweeps; results
            # that converge within 80 sweeps are unchanged by the larger cap.
            "max_sweeps": int(self.config["repair_sweeps"]),
            "active_row_budget": int(self.audited_config["repair_active_row_budget"]),
            "step": float(self.audited_config["repair_step"]),
            "osqp_max_iter": int(self.audited_config["osqp_max_iter"]),
        }

    def bundles(self, tolerance: float) -> dict[str, Any]:
        key = float(tolerance)
        if key not in self._bundles:
            self._bundles[key] = closure.build_dataset_bundles(
                self.assets_all, self.relations_all, self.policy_all, tolerance=key
            )
        return self._bundles[key]

    def add_check(
        self,
        stage: str,
        check: str,
        passed: bool,
        *,
        severity: str = "fatal",
        detail: str = "",
    ) -> None:
        self.quality = [
            row for row in self.quality if not (row["stage"] == stage and row["check"] == check)
        ]
        self.quality.append(
            {
                "stage": stage,
                "check": check,
                "severity": severity,
                "passed": bool(passed),
                "detail": str(detail),
            }
        )


def load_audited_frames(package_dir: Path) -> dict[str, pd.DataFrame]:
    package_dir = Path(package_dir)
    return {
        "assets": read_csv(package_dir / "00_workload" / "sql_assets.csv"),
        "relations": read_csv(package_dir / "00_workload" / "compiled_relations.csv"),
        "policy": read_csv(package_dir / "01_proposals_attacks" / "controlled_policy.csv"),
        "oof": read_csv(package_dir / "01_proposals_attacks" / "oof_predictions.csv"),
        "pooled": read_csv(package_dir / "01_proposals_attacks" / "model_metrics_pooled.csv"),
        "attacks": read_csv(package_dir / "01_proposals_attacks" / "certified_row_attacks.csv"),
        "gap": read_csv(package_dir / "01_proposals_attacks" / "omitted_relation_gap_cases.csv"),
    }


def build_context(
    *,
    mode: str,
    config: dict[str, Any],
    package_dir: Path,
    output_dir: Path,
    input_audit: pd.DataFrame,
    package_state_before: dict[str, str],
    session_id: str,
    lock: RunLock | None,
    log_fn: Callable[[str], None] = print,
) -> RevisionContext:
    package_dir = Path(package_dir)
    output_dir = Path(output_dir)
    audited_config = json.loads((package_dir / "run_config.json").read_text(encoding="utf-8"))
    if audited_config.get("suite_version") != AUDITED_SUITE_VERSION:
        raise RuntimeError("input package is not the audited-v4 suite")
    frames = load_audited_frames(package_dir)
    assets = frames["assets"]
    supported = assets[as_bool(assets["expected_supported"].astype(str))].copy()
    supported["asset_id"] = supported["asset_id"].astype(str)
    relations = frames["relations"].copy()
    policy = frames["policy"].copy()
    policy["asset_id"] = policy["asset_id"].astype(str)
    oof = frames["oof"].copy()
    oof["asset_id"] = oof["asset_id"].astype(str)
    features_all = add_structural_features(supported)
    config_digest = canonical_digest(config)
    checkpointer = Checkpointer(
        output_dir / "parts",
        mode=mode,
        input_digest=AUDITED_MANIFEST_SHA256,
        config_digest=config_digest,
        session_id=session_id,
    )
    seeds = set(int(seed) for seed in config["seeds"])
    datasets: dict[str, DatasetState] = {}
    for dataset, dataset_assets in supported.groupby("dataset", sort=False):
        dataset = str(dataset)
        ordered_ids = dataset_assets["asset_id"].astype(str).tolist()
        dataset_relations = relations[relations["dataset"] == dataset].reset_index(drop=True)
        snapshot = core.build_snapshot(ordered_ids, dataset_relations, epoch=1)
        price_map = policy[policy["dataset"] == dataset].set_index("asset_id")["target_price"]
        if set(price_map.index) != set(ordered_ids):
            raise RuntimeError(f"controlled price coverage mismatch for {dataset}")
        list_prices = price_map.loc[ordered_ids].to_numpy(dtype=np.float64)
        ordered_features = (
            features_all[features_all["dataset"] == dataset]
            .set_index("asset_id")
            .loc[ordered_ids]
            .reset_index()
        )
        proposals: dict[tuple[str, int], np.ndarray] = {}
        dataset_oof = oof[oof["dataset"] == dataset]
        for (model, seed), group in dataset_oof.groupby(["model", "seed"], sort=True):
            if int(seed) not in seeds:
                continue
            mapping = group.set_index("asset_id")["oof_prediction"]
            if set(mapping.index) != set(ordered_ids):
                raise RuntimeError(f"OOF coverage mismatch for {dataset}/{model}/{seed}")
            proposals[(str(model), int(seed))] = mapping.loc[ordered_ids].to_numpy(dtype=np.float64)
        expected = {(model, seed) for model in MODEL_NAMES for seed in seeds}
        if set(proposals) != expected:
            raise RuntimeError(f"missing audited proposals for {dataset}")
        datasets[dataset] = DatasetState(
            name=dataset,
            key=slug(dataset),
            assets=dataset_assets.reset_index(drop=True),
            asset_ids=ordered_ids,
            relations=dataset_relations,
            policy_frame=policy[policy["dataset"] == dataset][["dataset", "asset_id", "target_price"]]
            .set_index("asset_id")
            .loc[ordered_ids]
            .reset_index()[["dataset", "asset_id", "target_price"]],
            list_prices=list_prices,
            features=ordered_features,
            snapshot=snapshot,
            evaluator=RelationEvaluator(snapshot, list_prices),
            proposals=proposals,
        )
    if len(datasets) != 3:
        raise RuntimeError(f"expected three audited datasets, found {len(datasets)}")
    context = RevisionContext(
        mode=mode,
        config=config,
        audited_config=audited_config,
        package_dir=package_dir,
        output_dir=output_dir,
        checkpointer=checkpointer,
        lock=lock,
        session_id=session_id,
        datasets=datasets,
        assets_all=supported,
        relations_all=relations,
        policy_all=policy,
        oof_all=oof,
        pooled_metrics_all=frames["pooled"],
        audited_attacks=frames["attacks"],
        audited_gap=frames["gap"],
        input_audit=input_audit,
        package_state_before=package_state_before,
        log_fn=log_fn,
    )
    for state in datasets.values():
        verification = core.independent_verify(
            state.snapshot, state.list_prices, tolerance=PRIMARY_TOLERANCE
        )
        reproduced = raw_controlled_policy(state.features)
        context.add_check(
            "setup",
            f"{state.name}: audited controlled policy verifies at 1e-12",
            verification.passed,
            detail=f"max residual {verification.max_residual:.3e}",
        )
        context.add_check(
            "setup",
            f"{state.name}: audited policy equals the linear price function",
            bool(np.array_equal(reproduced, state.list_prices)),
            detail=f"max abs difference {float(np.max(np.abs(reproduced - state.list_prices))):.3e}",
        )
    failed = [row for row in context.quality if not row["passed"]]
    if failed:
        raise RuntimeError("Setup checks failed: " + "; ".join(row["check"] for row in failed))
    return context


# ---------------------------------------------------------------------------
# Running methods
# ---------------------------------------------------------------------------


def run_method(
    snapshot: core.ConstraintSnapshot,
    candidate: np.ndarray,
    method: str,
    *,
    tolerance: float,
    gate_options: Mapping[str, Any],
    active_row_budget: int | None = None,
    max_sweeps: int | None = None,
) -> dict[str, Any]:
    """Run one enforcement method and verify its output at ``tolerance``.

    ``osqp_qp_v4_stop`` is a diagnostic: OSQP with the audited stopping rule
    (the solver tolerance the audited code derives from 1e-7, eps_abs = 1e-9),
    followed by independent verification at ``tolerance``. Every other method
    is the audited implementation called with ``tolerance``.
    """

    options = dict(gate_options)
    if active_row_budget is not None:
        options["active_row_budget"] = int(active_row_budget)
    if max_sweeps is not None:
        options["max_sweeps"] = int(max_sweeps)
    vector = np.asarray(candidate, dtype=np.float64)
    if method == "osqp_qp_v4_stop":
        result = core.run_enforcement_method(
            snapshot, vector, method="osqp_qp", tolerance=AUDITED_TOLERANCE,
            allow_fallback=False, **options,
        )
        started = time.perf_counter_ns()
        verification = core.independent_verify(snapshot, result.prices, tolerance=tolerance)
        extra_ms = (time.perf_counter_ns() - started) / 1e6
        runtime_ms = result.runtime_ms + extra_ms
    elif method in {"reject_only", "arbiter_active", "full_cyclic", "osqp_qp"}:
        result = core.run_enforcement_method(
            snapshot, vector, method=method, tolerance=tolerance,
            allow_fallback=(method == "arbiter_active"), **options,
        )
        verification = result.verification
        runtime_ms = result.runtime_ms
    else:
        raise ValueError(f"unknown method: {method}")
    rejected = not verification.passed
    oracle = core.independent_verify(snapshot, result.prices, tolerance=tolerance)
    return {
        "method": method,
        "runtime_ms": float(runtime_ms),
        "verified": bool(verification.passed),
        "rejected": bool(rejected),
        "unsafe_release": bool((not rejected) and (not oracle.passed)),
        "fallback_used": bool(result.fallback_used),
        "solver": str(result.solver),
        "sweeps": int(result.sweeps),
        "projected_rows": int(result.projected_rows),
        "l1_adjustment": float(np.abs(result.prices - vector).sum()),
        "violations_after": int(oracle.violations),
        "max_residual_after": float(oracle.max_residual),
        "error": str(result.error),
        "_prices": result.prices,
    }


def gate(
    snapshot: core.ConstraintSnapshot,
    candidate: np.ndarray,
    *,
    tolerance: float,
    gate_options: Mapping[str, Any],
    active_row_budget: int | None = None,
):
    """The audited release gate: active-row repair, OSQP fallback, rejection."""

    options = dict(gate_options)
    if active_row_budget is not None:
        options["active_row_budget"] = int(active_row_budget)
    return core.run_enforcement_method(
        snapshot,
        np.asarray(candidate, dtype=np.float64),
        method="arbiter_active",
        tolerance=tolerance,
        allow_fallback=True,
        **options,
    )


def sweeps_needed(
    ctx: RevisionContext,
    snapshot: core.ConstraintSnapshot,
    candidate: np.ndarray,
    result: Any,
    *,
    tolerance: float = PRIMARY_TOLERANCE,
) -> dict[str, Any]:
    """Sweeps active-row repair alone needs to verify (fallback disabled).

    When the gate did not fall back, its own sweep count is the answer. When it
    did, active repair is rerun without fallback up to sweeps_to_verify_cap, so
    the fallback rate at any other sweep budget can be read off the data.
    """

    if not result.fallback_used and result.verification.passed:
        return {"sweeps_to_verify": int(result.sweeps), "verified_within_cap": True}
    options = {**ctx.gate_options(), "max_sweeps": int(ctx.config["sweeps_to_verify_cap"])}
    probe = core.run_enforcement_method(
        snapshot, np.asarray(candidate, dtype=np.float64), method="arbiter_active",
        tolerance=tolerance, allow_fallback=False, **options,
    )
    return {"sweeps_to_verify": int(probe.sweeps), "verified_within_cap": bool(probe.verification.passed)}


def run_parts(
    ctx: RevisionContext,
    stage: str,
    parts: Sequence[tuple[str, Callable[[], dict[str, pd.DataFrame]]]],
) -> dict[str, pd.DataFrame]:
    collected: dict[str, list[pd.DataFrame]] = defaultdict(list)
    for part_name, function in parts:
        part = f"{stage}/{part_name}"
        if ctx.checkpointer.done(part):
            ctx.log(f"{stage} {part_name}: reloaded from checkpoint")
        else:
            ctx.beat(part)
            started = time.perf_counter()
            frames = function()
            elapsed = time.perf_counter() - started
            ctx.checkpointer.save(part, frames, elapsed_s=elapsed)
            ctx.log(f"{stage} {part_name}: finished in {elapsed:.1f} s")
            ctx.beat(part)
        for name, frame in ctx.checkpointer.load(part).items():
            collected[name].append(frame)
    combined = {}
    for name, frames in collected.items():
        with_columns = [frame for frame in frames if len(frame.columns)]
        with_rows = [frame for frame in with_columns if len(frame)]
        if with_rows:
            combined[name] = pd.concat(with_rows, ignore_index=True, sort=False)
        elif with_columns:
            combined[name] = with_columns[0].iloc[0:0].copy()
        else:
            combined[name] = pd.DataFrame()
    return combined


def write_outputs(ctx: RevisionContext, stage: str, frames: Mapping[str, pd.DataFrame]) -> None:
    ctx.outputs_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        atomic_write_csv(ctx.outputs_dir / f"{stage}_{name}.csv", frame)
    ctx.stage_outputs[stage] = dict(frames)


# ---------------------------------------------------------------------------
# Stage 01: natural proposals
# ---------------------------------------------------------------------------


def _s01_tolerance_part(ctx: RevisionContext, state: DatasetState) -> dict[str, pd.DataFrame]:
    rows, tops = [], []
    options = ctx.gate_options()
    for (model, seed), proposal in state.ordered_proposals():
        for tolerance in ctx.config["natural_tolerances"]:
            tolerance = float(tolerance)
            before = core.independent_verify(state.snapshot, proposal, tolerance=tolerance)
            result = gate(state.snapshot, proposal, tolerance=tolerance, gate_options=options)
            released = not result.rejected
            oracle = core.independent_verify(state.snapshot, result.prices, tolerance=tolerance)
            primary = core.independent_verify(
                state.snapshot, result.prices, tolerance=PRIMARY_TOLERANCE
            )
            metrics = state.evaluator.metrics(result.prices) if released else {}
            needed = sweeps_needed(ctx, state.snapshot, proposal, result, tolerance=tolerance)
            rows.append(
                {
                    "dataset": state.name,
                    "model": model,
                    "seed": int(seed),
                    "tolerance": tolerance,
                    **needed,
                    "violations_before": before.violations,
                    "max_residual_before": before.max_residual,
                    "released": released,
                    "rejected": bool(result.rejected),
                    "fallback_used": bool(result.fallback_used),
                    "solver": result.solver,
                    "sweeps": int(result.sweeps),
                    "projected_rows": int(result.projected_rows),
                    "gate_ms": float(result.runtime_ms),
                    "fallback_ms": float(result.fallback_ms),
                    "l1_adjustment": float(result.l1_adjustment),
                    "violations_after": int(oracle.violations),
                    "max_residual_after": float(oracle.max_residual),
                    "unsafe_release": bool(released and not oracle.passed),
                    "violations_after_at_1e12": int(primary.violations),
                    "max_residual_after_at_1e12": float(primary.max_residual),
                    "relations_evaluated": metrics.get("relations_evaluated", 0),
                    "positive_residual_relations": metrics.get("positive_residual_relations", 0),
                    "relations_above_1e12": metrics.get("relations_above_1e12", 0),
                    "max_relation_residual": metrics.get("max_relation_residual", math.nan),
                    "max_saving_vs_list": metrics.get("max_saving_vs_list", math.nan),
                    "n_saving_gt_1pct": metrics.get("n_saving_gt_1pct", 0),
                    "n_saving_gt_0p1pct": metrics.get("n_saving_gt_0p1pct", 0),
                    "max_saving_conservative": metrics.get("max_saving_conservative", math.nan),
                    "n_conservative_gt_1pct": metrics.get("n_conservative_gt_1pct", 0),
                    "n_conservative_gt_0p1pct": metrics.get("n_conservative_gt_0p1pct", 0),
                    "argmax_relation_id": metrics.get("argmax_relation_id", ""),
                    "argmax_relation_type": metrics.get("argmax_relation_type", ""),
                    "price_digest": core.price_vector_digest(state.snapshot.asset_ids, result.prices),
                    "error": result.error,
                }
            )
            if released:
                for record in state.evaluator.top_relations(result.prices, k=10):
                    tops.append(
                        {
                            "dataset": state.name,
                            "model": model,
                            "seed": int(seed),
                            "tolerance": tolerance,
                            **record,
                        }
                    )
            ctx.beat("s01 tolerance")
    top_columns = [
        "dataset", "model", "seed", "tolerance", "rank", "relation_id", "relation_type",
        "residual", "target_list_price", "target_posted_price", "saving_vs_list",
    ]
    return {
        "natural_gate": pd.DataFrame(rows),
        "natural_top_relations": pd.DataFrame(tops, columns=top_columns),
    }


def _s01_heavy_part(ctx: RevisionContext, state: DatasetState) -> dict[str, pd.DataFrame]:
    methods = list(ctx.config["heavy_methods"])
    options = ctx.gate_options()
    rows = []
    for proposal_index, ((model, seed), proposal) in enumerate(state.ordered_proposals()):
        before = core.independent_verify(state.snapshot, proposal, tolerance=PRIMARY_TOLERANCE)
        for warmup in range(int(ctx.config["heavy_warmups"])):
            for method in rotate(methods, proposal_index + warmup):
                run_method(
                    state.snapshot, proposal, method,
                    tolerance=PRIMARY_TOLERANCE, gate_options=options,
                )
        for repetition in range(int(ctx.config["heavy_repetitions"])):
            ctx.beat("s01 heavy")
            order = rotate(methods, proposal_index + repetition)
            for execution_order, method in enumerate(order):
                outcome = run_method(
                    state.snapshot, proposal, method,
                    tolerance=PRIMARY_TOLERANCE, gate_options=options,
                )
                outcome.pop("_prices")
                rows.append(
                    {
                        "dataset": state.name,
                        "model": model,
                        "seed": int(seed),
                        "proposal_index": int(proposal_index),
                        "repetition": int(repetition),
                        "execution_order": int(execution_order),
                        "violations_before": int(before.violations),
                        **outcome,
                    }
                )
            ctx.beat("s01 heavy")
    return {"heavy_raw": pd.DataFrame(rows)}


def summarize_natural(natural: pd.DataFrame) -> pd.DataFrame:
    frame = natural.copy()
    for column in ["released", "rejected", "fallback_used", "unsafe_release"]:
        frame[column] = as_bool(frame[column])
    return (
        frame.groupby(["dataset", "model", "tolerance"], as_index=False)
        .agg(
            proposals=("seed", "size"),
            violations_before_min=("violations_before", "min"),
            violations_before_max=("violations_before", "max"),
            released_rate=("released", "mean"),
            fallback_rate=("fallback_used", "mean"),
            rejected_rate=("rejected", "mean"),
            unsafe_releases=("unsafe_release", "sum"),
            sweeps_max=("sweeps", "max"),
            gate_p50_ms=("gate_ms", "median"),
            gate_max_ms=("gate_ms", "max"),
            max_residual_after=("max_residual_after", "max"),
            max_saving_vs_list=("max_saving_vs_list", "max"),
            releases_with_saving_gt_1pct=("n_saving_gt_1pct", lambda values: int((values > 0).sum())),
            releases_with_saving_gt_0p1pct=("n_saving_gt_0p1pct", lambda values: int((values > 0).sum())),
            max_saving_conservative=("max_saving_conservative", "max"),
            releases_with_conservative_gt_1pct=("n_conservative_gt_1pct", lambda values: int((values > 0).sum())),
        )
        .sort_values(["dataset", "model", "tolerance"])
        .reset_index(drop=True)
    )


def summarize_tolerance_overall(natural: pd.DataFrame) -> pd.DataFrame:
    frame = natural.copy()
    for column in ["released", "rejected", "fallback_used", "unsafe_release"]:
        frame[column] = as_bool(frame[column])
    return (
        frame.groupby(["tolerance"], as_index=False)
        .agg(
            releases=("released", "sum"),
            proposals=("released", "size"),
            fallback=("fallback_used", "sum"),
            rejected=("rejected", "sum"),
            unsafe=("unsafe_release", "sum"),
            sweeps_max=("sweeps", "max"),
            max_saving_vs_list=("max_saving_vs_list", "max"),
            releases_with_saving_gt_1pct=("n_saving_gt_1pct", lambda values: int((values > 0).sum())),
            releases_with_saving_gt_0p1pct=("n_saving_gt_0p1pct", lambda values: int((values > 0).sum())),
            max_saving_conservative=("max_saving_conservative", "max"),
            releases_with_conservative_gt_1pct=("n_conservative_gt_1pct", lambda values: int((values > 0).sum())),
            max_residual_after=("max_residual_after", "max"),
            gate_p50_ms=("gate_ms", "median"),
        )
        .sort_values("tolerance", ascending=False)
        .reset_index(drop=True)
    )


def summarize_method_trials(
    raw: pd.DataFrame, group_columns: Sequence[str]
) -> pd.DataFrame:
    frame = raw.copy()
    for column in ["verified", "rejected", "unsafe_release", "fallback_used"]:
        if column in frame:
            frame[column] = as_bool(frame[column])
    return (
        frame.groupby(list(group_columns), as_index=False)
        .agg(
            trials=("runtime_ms", "size"),
            runtime_p50_ms=("runtime_ms", "median"),
            runtime_p95_ms=("runtime_ms", q95),
            runtime_p99_ms=("runtime_ms", q99),
            runtime_mean_ms=("runtime_ms", "mean"),
            verified_rate=("verified", "mean"),
            rejected_rate=("rejected", "mean"),
            fallback_rate=("fallback_used", "mean"),
            unsafe_releases=("unsafe_release", "sum"),
            sweeps_median=("sweeps", "median"),
            sweeps_max=("sweeps", "max"),
            max_residual_after=("max_residual_after", "max"),
            mean_l1_adjustment=("l1_adjustment", "mean"),
        )
        .reset_index(drop=True)
    )


def paired_speedups(
    raw: pd.DataFrame,
    *,
    cell_columns: Sequence[str],
    unit_column: str,
    repetition_column: str,
    reference: str,
    baselines: Sequence[str],
    resamples: int,
) -> pd.DataFrame:
    """Median of per-unit median ratios baseline/reference with bootstrap CIs.

    Units (attack cases or proposals) are the resampled clusters; timing
    repetitions are not treated as independent units.
    """

    frame = raw.copy()
    frame["verified"] = as_bool(frame["verified"])
    index = list(cell_columns) + [unit_column, repetition_column]
    runtime = frame.pivot_table(index=index, columns="method", values="runtime_ms", aggfunc="first")
    verified = frame.pivot_table(index=index, columns="method", values="verified", aggfunc="first")
    rows = []
    cells = runtime.reset_index().groupby(list(cell_columns), sort=True)
    for cell_keys, cell in cells:
        cell_keys = cell_keys if isinstance(cell_keys, tuple) else (cell_keys,)
        cell_index = pd.MultiIndex.from_frame(cell[index])
        for baseline in baselines:
            if baseline not in runtime.columns or reference not in runtime.columns:
                continue
            ratios = (
                runtime.loc[cell_index, baseline] / runtime.loc[cell_index, reference]
            ).rename("ratio").reset_index()
            ran = verified.loc[cell_index, baseline].notna().to_numpy()
            ratios = ratios[ran].reset_index(drop=True)
            baseline_verified = verified.loc[cell_index, baseline][ran].astype(bool).to_numpy()
            ratios["baseline_verified"] = baseline_verified
            for subset_label, subset in [
                ("all_trials", ratios),
                ("baseline_verified_trials", ratios[ratios["baseline_verified"]]),
            ]:
                unit_medians = subset.groupby(unit_column)["ratio"].median().to_numpy(float)
                if unit_medians.size:
                    median = float(np.median(unit_medians))
                    low, high = bootstrap_median_ci(
                        unit_medians,
                        resamples=resamples,
                        seed=stable_seed("paired", *cell_keys, baseline, subset_label),
                    )
                else:
                    median = low = high = math.nan
                rows.append(
                    {
                        **dict(zip(cell_columns, cell_keys)),
                        "baseline": baseline,
                        "subset": subset_label,
                        "paired_trials": int(len(subset)),
                        "units": int(unit_medians.size),
                        "baseline_verified_rate": float(np.mean(baseline_verified))
                        if baseline_verified.size
                        else math.nan,
                        "speedup_median": median,
                        "speedup_ci95_low": low,
                        "speedup_ci95_high": high,
                        "direction": f"{baseline} latency divided by {reference} latency",
                    }
                )
    return pd.DataFrame(rows)


def run_stage_01(ctx: RevisionContext) -> dict[str, pd.DataFrame]:
    parts = []
    for state in ctx.datasets.values():
        parts.append((f"{state.key}/tolerance", lambda state=state: _s01_tolerance_part(ctx, state)))
    for state in ctx.datasets.values():
        parts.append((f"{state.key}/heavy", lambda state=state: _s01_heavy_part(ctx, state)))
    frames = run_parts(ctx, "s01", parts)
    natural = frames["natural_gate"]
    heavy = frames["heavy_raw"]
    outputs = {
        "natural_gate": natural,
        "natural_top_relations": frames["natural_top_relations"],
        "natural_summary": summarize_natural(natural),
        "tolerance_overall": summarize_tolerance_overall(natural),
        "heavy_raw": heavy,
        "heavy_summary": summarize_method_trials(heavy, ["dataset", "method"]),
        "heavy_paired": paired_speedups(
            heavy,
            cell_columns=["dataset"],
            unit_column="proposal_index",
            repetition_column="repetition",
            reference="arbiter_active",
            baselines=list(BASELINE_METHODS) + ["reject_only"],
            resamples=int(ctx.config["bootstrap_resamples"]),
        ),
    }
    write_outputs(ctx, "s01", outputs)
    expected_rows = len(ctx.datasets) * len(MODEL_NAMES) * len(ctx.config["seeds"]) * len(
        ctx.config["natural_tolerances"]
    )
    released = as_bool(natural["released"])
    ctx.add_check("s01", "one natural-gate row per proposal and tolerance", len(natural) == expected_rows,
                  detail=f"{len(natural)}/{expected_rows}")
    ctx.add_check("s01", "no natural proposal release fails independent verification",
                  not as_bool(natural["unsafe_release"]).any())
    primary = natural[natural["tolerance"] == PRIMARY_TOLERANCE]
    ctx.add_check(
        "s01",
        "natural proposals released at 1e-12 pass the complete snapshot at 1e-12",
        bool((primary.loc[as_bool(primary["released"]), "violations_after_at_1e12"] == 0).all()),
    )
    ctx.add_check(
        "s01",
        "all natural proposals release at 1e-12 (rejections are recorded, not hidden)",
        bool(as_bool(primary["released"]).all()),
        severity="finding",
        detail=f"{int(as_bool(primary['released']).sum())}/{len(primary)} released",
    )
    expected_heavy = (
        len(ctx.datasets) * len(MODEL_NAMES) * len(ctx.config["seeds"])
        * int(ctx.config["heavy_repetitions"]) * len(ctx.config["heavy_methods"])
    )
    ctx.add_check("s01", "heavy-load grid is complete", len(heavy) == expected_heavy,
                  detail=f"{len(heavy)}/{expected_heavy}")
    ctx.add_check("s01", "heavy-load trials contain no unsafe release",
                  not as_bool(heavy["unsafe_release"]).any())
    ctx.add_check(
        "s01",
        "OSQP trials used the OSQP backend, not the SciPy fallback",
        not heavy["solver"].astype(str).str.contains("scipy").any(),
    )
    return outputs


# ---------------------------------------------------------------------------
# Stage 02: exact per-row box attacks
# ---------------------------------------------------------------------------


def _attack_summaries(attacks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = attacks.copy()
    for column in ["released", "fallback_used", "unsafe_release"]:
        frame[column] = as_bool(frame[column])
    common = dict(
        attacks=("attack_id", "size"),
        distinct_target_rows=("target_row_id", "nunique"),
        attack_success_rate=("violations_before", lambda values: float((values > 0).mean())),
        violations_before_max=("violations_before", "max"),
        released_rate=("released", "mean"),
        fallback_rate=("fallback_used", "mean"),
        unsafe_release_rate=("unsafe_release", "mean"),
        gate_p50_ms=("repair_ms", "median"),
        gate_p95_ms=("repair_ms", q95),
        l1_median=("l1_adjustment", "median"),
        max_post_residual=("max_residual_after", "max"),
        sweeps_max=("sweeps", "max"),
    )
    summary = frame.groupby(["dataset", "budget_fraction"], as_index=False).agg(**common)
    family = frame.groupby(
        ["dataset", "budget_fraction", "target_constraint_type"], as_index=False
    ).agg(**common)
    return summary, family


def compare_attacks_with_audited(new: pd.DataFrame, old: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "budget_fraction", "target_constraint_type", "target_row_id"]
    cells = ["dataset", "budget_fraction", "target_constraint_type"]
    present = set(new[cells].itertuples(index=False, name=None))
    old = old[[tuple(row) in present for row in old[cells].itertuples(index=False, name=None)]]
    left = new[keys + ["released", "violations_before", "sweeps"]].copy()
    right = old[keys + ["released", "violations_before", "sweeps"]].copy()
    left["released"] = as_bool(left["released"])
    right["released"] = as_bool(right["released"])
    merged = left.merge(right, on=keys, how="outer", suffixes=("_1e12", "_v4"), indicator=True)
    rows = []
    for cell, group in merged.groupby(["dataset", "budget_fraction", "target_constraint_type"], sort=True):
        both = group[group["_merge"] == "both"]
        rows.append(
            {
                "dataset": cell[0],
                "budget_fraction": cell[1],
                "target_constraint_type": cell[2],
                "cases_1e12": int((group["_merge"] != "right_only").sum()),
                "cases_v4": int((group["_merge"] != "left_only").sum()),
                "same_target_rows": int(len(both)),
                "same_release_decision": int((both["released_1e12"] == both["released_v4"]).sum()),
                "same_violations_before": int(
                    (both["violations_before_1e12"] == both["violations_before_v4"]).sum()
                ),
                "same_sweeps": int((both["sweeps_1e12"] == both["sweeps_v4"]).sum()),
            }
        )
    return pd.DataFrame(rows)


def run_stage_02(ctx: RevisionContext) -> dict[str, pd.DataFrame]:
    options = ctx.gate_options()

    def part(state: DatasetState) -> dict[str, pd.DataFrame]:
        attacks, records, deltas = run_certified_attack_campaign(
            assets=state.assets,
            relations=state.relations,
            policy=state.policy_frame,
            budgets=[float(value) for value in ctx.config["attack_budgets"]],
            requested_cases_per_relation_type=int(ctx.config["attack_cases"]),
            minimum_cases_per_relation_type=int(ctx.config["attack_min_cases"]),
            tolerance=PRIMARY_TOLERANCE,
            repair_options=options,
        )
        delta_columns = ["attack_id", "asset_id", "candidate_price", "released_price", "delta"]
        return {
            "attacks": attacks,
            "attack_records": records,
            "attack_deltas": deltas if len(deltas.columns) else pd.DataFrame(columns=delta_columns),
        }

    frames = run_parts(ctx, "s02", [(state.key, lambda state=state: part(state)) for state in ctx.datasets.values()])
    attacks = frames["attacks"]
    summary, family = _attack_summaries(attacks)
    comparison = compare_attacks_with_audited(attacks, ctx.audited_attacks)
    outputs = {
        "attacks": attacks,
        "attack_records": frames["attack_records"],
        "attack_deltas": frames["attack_deltas"],
        "attack_summary": summary,
        "attack_family_summary": family,
        "attack_v4_comparison": comparison,
    }
    write_outputs(ctx, "s02", outputs)
    expected_cells = {
        (state.name, float(budget), family_name)
        for state in ctx.datasets.values()
        for budget in ctx.config["attack_budgets"]
        for family_name in RELATION_FAMILIES
    }
    observed_cells = set(
        attacks[["dataset", "budget_fraction", "target_constraint_type"]].itertuples(index=False, name=None)
    )
    ctx.add_check("s02", "every dataset, budget, and relation family is attacked",
                  observed_cells == expected_cells,
                  detail=f"{len(observed_cells)}/{len(expected_cells)} cells")
    ctx.add_check("s02", "attack target rows are unique within each cell",
                  not attacks.duplicated(["dataset", "budget_fraction", "target_constraint_type", "target_row_id"]).any())
    ctx.add_check("s02", "no attack release fails independent verification",
                  not as_bool(attacks["unsafe_release"]).any())
    released = attacks[as_bool(attacks["released"])]
    ctx.add_check("s02", "released attack vectors have zero violated rows at 1e-12",
                  bool((released["violations_after"] == 0).all()))
    ctx.add_check("s02", "every attack is repaired and released at 1e-12",
                  bool(as_bool(attacks["released"]).all()), severity="finding",
                  detail=f"{len(released)}/{len(attacks)}")
    return outputs


# ---------------------------------------------------------------------------
# Stage 03: omitted-relation gap
# ---------------------------------------------------------------------------


def run_stage_03(ctx: RevisionContext) -> dict[str, pd.DataFrame]:
    options = ctx.gate_options()

    def part(state: DatasetState) -> dict[str, pd.DataFrame]:
        cases = run_omitted_relation_gap_campaign(
            assets=state.assets,
            relations=state.relations,
            policy=state.policy_frame,
            budgets=[float(value) for value in ctx.config["attack_budgets"]],
            cases_per_relation_type=int(ctx.config["gap_cases"]),
            tolerance=PRIMARY_TOLERANCE,
            repair_options=options,
        )
        return {"gap_cases": cases}

    frames = run_parts(ctx, "s03", [(state.key, lambda state=state: part(state)) for state in ctx.datasets.values()])
    cases = frames["gap_cases"].copy()
    for column in ["released_by_incomplete_gate", "gap_survives", "fallback_used"]:
        cases[column] = as_bool(cases[column])
    summary = (
        cases.groupby(["dataset", "budget_fraction", "relation_type"], as_index=False)
        .agg(
            cases=("relation_id", "size"),
            distinct_relations=("relation_id", "nunique"),
            candidate_target_violation_rate=(
                "certified_target_residual",
                lambda values: float((values > PRIMARY_TOLERANCE).mean()),
            ),
            incomplete_release_rate=("released_by_incomplete_gate", "mean"),
            gap_survival_rate=("gap_survives", "mean"),
            gaps_survived=("gap_survives", "sum"),
            fallback_rate=("fallback_used", "mean"),
            gate_p50_ms=("gate_ms", "median"),
        )
    )
    old = ctx.audited_gap.copy()
    old["gap_survives"] = as_bool(old["gap_survives"])
    merged = cases[["case_id", "gap_survives"]].merge(
        old[["case_id", "gap_survives"]], on="case_id", how="outer", suffixes=("_1e12", "_v4"), indicator=True
    )
    both = merged[merged["_merge"] == "both"]
    comparison = pd.DataFrame(
        [
            {
                "cases_1e12": int((merged["_merge"] != "right_only").sum()),
                "cases_v4": int((merged["_merge"] != "left_only").sum()),
                "same_case_ids": int(len(both)),
                "same_gap_outcome": int((both["gap_survives_1e12"] == both["gap_survives_v4"]).sum()),
                "gaps_survived_1e12": int(cases["gap_survives"].sum()),
                "gaps_survived_v4": int(old["gap_survives"].sum()),
            }
        ]
    )
    outputs = {"gap_cases": cases, "gap_summary": summary, "gap_v4_comparison": comparison}
    write_outputs(ctx, "s03", outputs)
    expected_cells = {
        (state.name, float(budget), family_name)
        for state in ctx.datasets.values()
        for budget in ctx.config["attack_budgets"]
        for family_name in RELATION_FAMILIES
    }
    observed = set(cases[["dataset", "budget_fraction", "relation_type"]].itertuples(index=False, name=None))
    ctx.add_check("s03", "omitted-row grid is complete", observed == expected_cells,
                  detail=f"{len(observed)}/{len(expected_cells)} cells")
    ctx.add_check(
        "s03",
        "incomplete-gate releases pass every encoded row",
        bool((cases.loc[cases["released_by_incomplete_gate"], "incomplete_violations_after"] == 0).all()),
    )
    return outputs


# ---------------------------------------------------------------------------
# Stage 04 and 05: scale benchmark and ablation
# ---------------------------------------------------------------------------


def run_scale_cell(
    ctx: RevisionContext,
    state: DatasetState,
    scale: int,
    *,
    methods: Sequence[str],
    repetitions: int,
    warmups: int,
    cases_per_family: int,
    attack_budget: float,
    tolerance: float,
    relations: pd.DataFrame | None = None,
    active_row_budget: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One dataset and scale of the audited scale benchmark, extended.

    The asset selection, attack construction, warm-up rotation, and paired
    trial order replicate arbiter_experiments.run_scale_benchmark. The
    extension adds the OSQP diagnostic method and per-trial solver errors.
    """

    options = ctx.gate_options()
    dataset_assets = state.assets.reset_index(drop=True)
    dataset_relations = state.relations if relations is None else relations.reset_index(drop=True)
    ordered_ids = balanced_relation_closed_asset_ids(dataset_assets, dataset_relations, int(scale))
    subset_relations = relation_subset(dataset_relations, ordered_ids)
    expected_types = set(dataset_relations["relation_type"].astype(str))
    observed_types = set(subset_relations["relation_type"].astype(str))
    if observed_types != expected_types:
        raise RuntimeError(
            f"scale subset lost relation families for {state.name}/{scale}: "
            f"expected {sorted(expected_types)}, observed {sorted(observed_types)}"
        )
    build_rows, build_times = [], []
    snapshot = None
    for build_repetition in range(int(repetitions)):
        started = time.perf_counter_ns()
        built = core.build_snapshot(ordered_ids, subset_relations, epoch=1)
        build_ms = (time.perf_counter_ns() - started) / 1e6
        build_times.append(build_ms)
        snapshot = built
        build_rows.append(
            {
                "dataset": state.name,
                "scale": int(scale),
                "repetition": int(build_repetition),
                "snapshot_build_ms": float(build_ms),
                "assets": built.assets,
                "constraints": built.constraints,
                "nonzeros": built.nonzeros,
                "density": built.density,
                "constraint_digest": built.constraint_digest,
            }
        )
    assert snapshot is not None
    price_map = dict(zip(state.asset_ids, state.list_prices))
    center = np.asarray([price_map[asset_id] for asset_id in ordered_ids], dtype=np.float64)
    if not core.independent_verify(snapshot, center, tolerance=tolerance).passed:
        raise RuntimeError(f"subset policy is infeasible for {state.name}/{scale}")
    relation_types = sorted(expected_types)
    centers = {
        family: relation_family_attack_center(snapshot, center, family, tolerance=tolerance)
        for family in relation_types
    }
    pools = {
        family: core.exact_box_row_attacks(
            snapshot,
            centers[family][0],
            budget_fraction=float(attack_budget),
            limit=int(cases_per_family),
            tolerance=tolerance,
            include_types=[family],
        )
        for family in relation_types
    }
    short = {family: len(pool) for family, pool in pools.items() if len(pool) < int(cases_per_family)}
    if short:
        raise RuntimeError(f"insufficient unique scale attacks for {state.name}/{scale}: {short}")
    attack_pool = [attack for family in relation_types for attack in pools[family]]
    method_list = list(methods)
    osqp_repetitions = int(ctx.config.get("osqp_qp_repetitions", repetitions))
    osqp_warmups = int(ctx.config.get("osqp_qp_warmups", warmups))

    def methods_for(index: int, limit: int) -> list[str]:
        return [m for m in method_list if m != "osqp_qp" or index < limit]

    rows = []
    for case_index, attack in enumerate(attack_pool):
        candidate = attack["candidate"]
        candidate_digest = hashlib.sha256(np.asarray(candidate, dtype="<f8").tobytes(order="C")).hexdigest()
        attack_case_id = hashlib.sha256(
            f"{state.name}|{scale}|{attack_budget}|{attack['constraint_type']}|{attack['row_id']}".encode("utf-8")
        ).hexdigest()[:24]
        precheck = core.independent_verify(snapshot, candidate, tolerance=tolerance)
        for warmup in range(int(warmups)):
            for method in rotate(methods_for(warmup, osqp_warmups), case_index):
                run_method(snapshot, candidate, method, tolerance=tolerance, gate_options=options,
                           active_row_budget=active_row_budget)
        for repetition in range(int(repetitions)):
            ctx.beat(f"scale {state.key} {scale}")
            order = rotate(methods_for(repetition, osqp_repetitions), case_index + repetition)
            for execution_order, method in enumerate(order):
                outcome = run_method(snapshot, candidate, method, tolerance=tolerance, gate_options=options,
                                     active_row_budget=active_row_budget)
                outcome.pop("_prices")
                build_ms = build_times[repetition % len(build_times)]
                rows.append(
                    {
                        "dataset": state.name,
                        "scale": int(scale),
                        "attack_case_id": attack_case_id,
                        "case_index": int(case_index),
                        "repetition": int(repetition),
                        "execution_order": int(execution_order),
                        "assets": snapshot.assets,
                        "constraints": snapshot.constraints,
                        "nonzeros": snapshot.nonzeros,
                        "density": snapshot.density,
                        "target_row_id": attack["row_id"],
                        "target_constraint_type": attack["constraint_type"],
                        "attack_center_policy": centers[attack["constraint_type"]][1],
                        "candidate_digest": candidate_digest,
                        "certified_target_residual": attack["certified_target_residual"],
                        "violations_before": int(precheck.violations),
                        "max_residual_before": float(precheck.max_residual),
                        "snapshot_build_ms": float(build_ms),
                        "snapshot_build_plus_gate_ms": float(build_ms + outcome["runtime_ms"]),
                        "constraint_digest": snapshot.constraint_digest,
                        **outcome,
                    }
                )
        ctx.beat(f"scale {state.key} {scale}")
    return pd.DataFrame(rows), pd.DataFrame(build_rows)


def run_stage_04(ctx: RevisionContext) -> dict[str, pd.DataFrame]:
    cfg = ctx.config

    def part(state: DatasetState, scale: int) -> dict[str, pd.DataFrame]:
        trials, builds = run_scale_cell(
            ctx, state, int(scale),
            methods=cfg["scale_methods"],
            repetitions=int(cfg["scale_repetitions"]),
            warmups=int(cfg["scale_warmups"]),
            cases_per_family=int(cfg["scale_cases_per_family"]),
            attack_budget=float(cfg["scale_attack_budget"]),
            tolerance=PRIMARY_TOLERANCE,
        )
        return {"scale_raw": trials, "build_raw": builds}

    parts = [
        (f"{state.key}/scale-{int(scale):05d}", lambda state=state, scale=scale: part(state, scale))
        for state in ctx.datasets.values()
        for scale in cfg["scale_points"]
    ]
    frames = run_parts(ctx, "s04", parts)
    raw = frames["scale_raw"]
    builds = frames["build_raw"]
    summary = summarize_method_trials(raw, ["dataset", "scale", "method"])
    build_summary = (
        builds.groupby(["dataset", "scale"], as_index=False)
        .agg(
            builds=("snapshot_build_ms", "size"),
            build_p50_ms=("snapshot_build_ms", "median"),
            build_p95_ms=("snapshot_build_ms", q95),
            constraints=("constraints", "first"),
            nonzeros=("nonzeros", "first"),
            density=("density", "first"),
        )
    )
    paired = paired_speedups(
        raw,
        cell_columns=["dataset", "scale"],
        unit_column="attack_case_id",
        repetition_column="repetition",
        reference="arbiter_active",
        baselines=list(BASELINE_METHODS) + ["reject_only"],
        resamples=int(cfg["bootstrap_resamples"]),
    )
    outputs = {
        "scale_raw": raw,
        "build_raw": builds,
        "scale_summary": summary,
        "build_summary": build_summary,
        "scale_paired": paired,
    }
    write_outputs(ctx, "s04", outputs)
    expected = {(state.name, int(scale)) for state in ctx.datasets.values() for scale in cfg["scale_points"]}
    observed = set(raw[["dataset", "scale"]].drop_duplicates().itertuples(index=False, name=None))
    ctx.add_check("s04", "every dataset and scale cell is present", observed == expected,
                  detail=f"{len(observed)}/{len(expected)}")
    per_trial = raw.groupby(["dataset", "scale", "attack_case_id", "repetition"])["method"].nunique().reset_index()
    osqp_in_design = "osqp_qp" in cfg["scale_methods"]
    expected_methods = np.where(
        per_trial["repetition"] < int(cfg.get("osqp_qp_repetitions", cfg["scale_repetitions"])),
        len(cfg["scale_methods"]),
        len(cfg["scale_methods"]) - int(osqp_in_design),
    )
    ctx.add_check("s04", "each candidate and repetition contains every scheduled method",
                  bool((per_trial["method"].to_numpy() == expected_methods).all()))
    consistency = raw.groupby(["dataset", "scale", "attack_case_id"]).agg(
        digests=("constraint_digest", "nunique"), candidates=("candidate_digest", "nunique")
    )
    ctx.add_check("s04", "all methods use the same snapshot and candidate",
                  bool(((consistency["digests"] == 1) & (consistency["candidates"] == 1)).all()))
    full_reps = raw[raw["repetition"] < int(cfg.get("osqp_qp_repetitions", cfg["scale_repetitions"]))]
    order_counts = full_reps.groupby(["dataset", "scale", "method", "execution_order"]).size()
    spread = order_counts.groupby(level=[0, 1, 2]).agg(lambda values: int(values.max() - values.min()))
    ctx.add_check("s04", "method execution order is counterbalanced", bool((spread <= 1).all()),
                  severity="warning")
    ctx.add_check("s04", "benchmark candidates violate the snapshot", bool((raw["violations_before"] > 0).all()))
    ctx.add_check("s04", "no scale trial releases an unverified vector", not as_bool(raw["unsafe_release"]).any())
    arbiter = raw[raw["method"] == "arbiter_active"]
    ctx.add_check("s04", "Arbiter verifies every scale candidate at 1e-12", bool(as_bool(arbiter["verified"]).all()),
                  severity="finding", detail=f"verified rate {as_bool(arbiter['verified']).mean():.3f}")
    ctx.add_check("s04", "OSQP trials used the OSQP backend, not the SciPy fallback",
                  not raw["solver"].astype(str).str.contains("scipy").any())
    return outputs


ABLATION_SPECS = (
    ("decomposition_only_budget_12000", ("decomposition",), 12000),
    ("all_relations_budget_64", RELATION_FAMILIES, 64),
    ("all_relations_budget_512", RELATION_FAMILIES, 512),
    ("all_relations_budget_12000", RELATION_FAMILIES, 12000),
)


def run_stage_05(ctx: RevisionContext) -> dict[str, pd.DataFrame]:
    cfg = ctx.config
    scale = min(int(cfg["ablation_scale"]), int(ctx.audited_config["max_assets_per_dataset"]))

    def part(state: DatasetState, label: str, families: Sequence[str], budget: int) -> dict[str, pd.DataFrame]:
        relations = state.relations[state.relations["relation_type"].isin(list(families))]
        trials, _ = run_scale_cell(
            ctx, state, scale,
            methods=["arbiter_active"],
            repetitions=int(cfg["ablation_repetitions"]),
            warmups=int(cfg["ablation_warmups"]),
            cases_per_family=int(cfg["scale_cases_per_family"]),
            attack_budget=float(cfg["scale_attack_budget"]),
            tolerance=PRIMARY_TOLERANCE,
            relations=relations,
            active_row_budget=int(budget),
        )
        trials.insert(0, "ablation", label)
        return {"ablation_raw": trials}

    parts = [
        (f"{state.key}/{label}", lambda state=state, label=label, families=families, budget=budget: part(state, label, families, budget))
        for state in ctx.datasets.values()
        for label, families, budget in ABLATION_SPECS
    ]
    frames = run_parts(ctx, "s05", parts)
    raw = frames["ablation_raw"]
    summary = summarize_method_trials(raw, ["dataset", "scale", "ablation"])
    outputs = {"ablation_raw": raw, "ablation_summary": summary}
    write_outputs(ctx, "s05", outputs)
    ctx.add_check("s05", "ablation grid is complete",
                  raw[["dataset", "ablation"]].drop_duplicates().shape[0] == len(ctx.datasets) * len(ABLATION_SPECS))
    ctx.add_check("s05", "no ablation trial releases an unverified vector", not as_bool(raw["unsafe_release"]).any())
    return outputs


# ---------------------------------------------------------------------------
# Stage 06 and 07: dense active-row budgets and fallback kernels
# ---------------------------------------------------------------------------


def run_optimized_tiered_fail_safe_repair(
    snapshot: core.ConstraintSnapshot,
    candidate: np.ndarray,
    *,
    tolerance: float,
    gate_options: Mapping[str, Any],
    active_row_budget: int,
    violated_only_sweeps: int,
) -> dict[str, Any]:
    """Active repair, then violated-only cyclic, then OSQP, then rejection.

    Same tiers as run_optimized_tiered_fail_safe_repair in the Notebook 06
    hardening module; reimplemented here so that a failed tier is recorded
    rather than raised.
    """

    options = dict(gate_options)
    original = np.asarray(candidate, dtype=np.float64)
    active = core.run_enforcement_method(
        snapshot, original, method="arbiter_active", tolerance=tolerance,
        allow_fallback=False, **{**options, "active_row_budget": int(active_row_budget)},
    )
    stage, prices, runtime, fallback_ms = "none", active.prices, active.runtime_ms, 0.0
    sweeps, projected = active.sweeps, active.projected_rows
    errors = [active.error] if active.error else []
    verification = active.verification
    if not verification.passed:
        violated_only = core.run_enforcement_method(
            snapshot, active.prices, method="arbiter_active", tolerance=tolerance,
            allow_fallback=False,
            **{**options, "active_row_budget": snapshot.constraints, "max_sweeps": int(violated_only_sweeps)},
        )
        runtime += violated_only.runtime_ms
        fallback_ms += violated_only.runtime_ms
        sweeps += violated_only.sweeps
        projected += violated_only.projected_rows
        if violated_only.error:
            errors.append(violated_only.error)
        stage, prices, verification = "violated_only_cyclic", violated_only.prices, violated_only.verification
        if not verification.passed:
            qp = core.run_enforcement_method(
                snapshot, original, method="osqp_qp", tolerance=tolerance,
                allow_fallback=False, **options,
            )
            runtime += qp.runtime_ms
            fallback_ms += qp.runtime_ms
            sweeps += qp.sweeps
            if qp.error:
                errors.append(qp.error)
            verification = core.independent_verify(snapshot, qp.prices, tolerance=tolerance)
            stage = "osqp" if verification.passed else "rejected_after_osqp"
            prices = qp.prices if verification.passed else original
    rejected = not verification.passed
    oracle = core.independent_verify(snapshot, prices, tolerance=tolerance)
    return {
        "fallback_used": stage != "none",
        "fallback_stage": stage,
        "rejected": bool(rejected),
        "verified": bool(verification.passed),
        "oracle_verified": bool(oracle.passed),
        "unsafe_release": bool((not rejected) and not oracle.passed),
        "runtime_ms": float(runtime),
        "fallback_ms": float(fallback_ms),
        "sweeps": int(sweeps),
        "projected_rows": int(projected),
        "violations_after": int(oracle.violations),
        "max_residual_after": float(oracle.max_residual),
        "l1_adjustment": float(np.abs(prices - original).sum()),
        "error": "; ".join(errors),
    }


def dense_cases_for(ctx: RevisionContext, tolerance: float, dataset: str) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    bundles = ctx.bundles(tolerance)
    return closure.build_dense_candidate_catalog(
        {dataset: bundles[dataset]},
        tolerance=float(tolerance),
        target_violation_counts=tuple(int(value) for value in ctx.config["dense_targets"]),
        budget_fraction=float(ctx.config["dense_budget_fraction"]),
    )


def run_stage_06(ctx: RevisionContext) -> dict[str, pd.DataFrame]:
    cfg = ctx.config
    options = ctx.gate_options()

    def part(tolerance: float, max_sweeps: int, state: DatasetState) -> dict[str, pd.DataFrame]:
        setting_options = {**options, "max_sweeps": int(max_sweeps)}
        cases, catalog = dense_cases_for(ctx, tolerance, state.name)
        raw, summary = closure.run_binding_budget_ablation(
            cases,
            budgets=[int(value) for value in cfg["dense_budgets"]],
            repetitions=int(cfg["dense_repetitions"]),
            warmups=int(cfg["dense_warmups"]),
            tolerance=float(tolerance),
            max_sweeps=int(max_sweeps),
            step=options["step"],
            osqp_max_iter=options["osqp_max_iter"],
        )
        optimized_rows = []
        for case in cases:
            precheck = core.independent_verify(case["_snapshot"], case["_candidate"], tolerance=tolerance)
            for budget in cfg["dense_budgets"]:
                for sweeps in cfg["fallback_sweeps_grid"]:
                    outcome = run_optimized_tiered_fail_safe_repair(
                        case["_snapshot"], case["_candidate"], tolerance=float(tolerance),
                        gate_options=setting_options, active_row_budget=int(budget),
                        violated_only_sweeps=int(sweeps),
                    )
                    optimized_rows.append(
                        {
                            "dataset": case["dataset"],
                            "relation_family": case["relation_family"],
                            "case_id": case["case_id"],
                            "active_row_budget": int(budget),
                            "max_sweeps": int(max_sweeps),
                            "violated_only_sweeps": int(sweeps),
                            "violations_before": int(precheck.violations),
                            "budget_binding": bool(precheck.violations > int(budget)),
                            **outcome,
                        }
                    )
            ctx.beat("s06 dense")
        for frame in (catalog, summary):
            frame.insert(0, "max_sweeps", int(max_sweeps))
        for frame in (catalog, raw, summary):
            frame.insert(0, "tolerance", float(tolerance))
        optimized = pd.DataFrame(optimized_rows)
        optimized.insert(0, "tolerance", float(tolerance))
        return {
            "dense_catalog": catalog,
            "dense_raw": raw,
            "dense_summary": summary,
            "dense_optimized": optimized,
        }

    parts = [
        (
            f"tol-{float(tolerance):.0e}-sweeps-{int(max_sweeps)}/{state.key}",
            lambda tolerance=tolerance, max_sweeps=max_sweeps, state=state: part(float(tolerance), int(max_sweeps), state),
        )
        for tolerance, max_sweeps in cfg["dense_settings"]
        for state in ctx.datasets.values()
    ]
    frames = run_parts(ctx, "s06", parts)
    raw = frames["dense_raw"].copy()
    for column in ["budget_binding", "fallback_used", "rejected", "oracle_verified", "unsafe_release"]:
        raw[column] = as_bool(raw[column])
    optimized = frames["dense_optimized"].copy()
    for column in ["budget_binding", "fallback_used", "rejected", "oracle_verified", "unsafe_release"]:
        optimized[column] = as_bool(optimized[column])
    budget_overview = (
        raw.groupby(["tolerance", "max_sweeps", "variant", "active_row_budget"], as_index=False)
        .agg(
            trials=("runtime_ms", "size"),
            cases=("case_id", "nunique"),
            binding_rate=("budget_binding", "mean"),
            fallback_rate=("fallback_used", "mean"),
            rejection_rate=("rejected", "mean"),
            verified_rate=("oracle_verified", "mean"),
            unsafe_releases=("unsafe_release", "sum"),
            runtime_p50_ms=("runtime_ms", "median"),
        )
    )
    optimized_overview = (
        optimized.groupby(["tolerance", "max_sweeps", "violated_only_sweeps", "active_row_budget"], as_index=False)
        .agg(
            cases=("case_id", "nunique"),
            fallback_rate=("fallback_used", "mean"),
            rejection_rate=("rejected", "mean"),
            verified_rate=("oracle_verified", "mean"),
            unsafe_releases=("unsafe_release", "sum"),
            runtime_p50_ms=("runtime_ms", "median"),
        )
    )
    stage_counts = (
        pd.concat(
            [
                raw.assign(variant_label=raw["variant"])[["tolerance", "max_sweeps", "variant_label", "fallback_stage"]],
                optimized.assign(
                    variant_label="optimized_fail_safe_" + optimized["violated_only_sweeps"].astype(int).astype(str) + "_sweeps"
                )[["tolerance", "max_sweeps", "variant_label", "fallback_stage"]],
            ],
            ignore_index=True,
        )
        .groupby(["tolerance", "max_sweeps", "variant_label", "fallback_stage"], as_index=False)
        .size()
        .rename(columns={"size": "trials"})
    )
    outputs = {
        "dense_catalog": frames["dense_catalog"],
        "dense_raw": raw,
        "dense_summary": frames["dense_summary"],
        "dense_optimized": optimized,
        "dense_budget_overview": budget_overview,
        "dense_optimized_overview": optimized_overview,
        "dense_fallback_stages": stage_counts,
    }
    write_outputs(ctx, "s06", outputs)
    fail_safe = raw[raw["variant"] == "fail_safe"]
    ctx.add_check("s06", "fail-safe variants release no unverified vector",
                  not fail_safe["unsafe_release"].any() and not optimized["unsafe_release"].any())
    ctx.add_check("s06", "active-only variant releases no unverified vector",
                  not raw.loc[raw["variant"] == "active_only", "unsafe_release"].any())
    binding = all(
        raw.loc[(raw["active_row_budget"] == budget), "budget_binding"].any()
        for budget in sorted(raw["active_row_budget"].unique())
    )
    ctx.add_check("s06", "every tested budget is binding in at least one dense case", binding, severity="warning")
    return outputs


def run_stage_07(ctx: RevisionContext) -> dict[str, pd.DataFrame]:
    cfg = ctx.config
    options = ctx.gate_options()
    dense_raw = ctx.stage_outputs.get("s06", {}).get("dense_raw")
    if dense_raw is None:
        dense_raw = read_csv(ctx.outputs_dir / "s06_dense_raw.csv")
    dense = dense_raw.copy()
    dense["fallback_used"] = as_bool(dense["fallback_used"])
    condition_frame = (
        dense[
            (dense["tolerance"] == PRIMARY_TOLERANCE)
            & (dense["max_sweeps"] == int(cfg["fallback_condition_sweeps"]))
            & (dense["variant"] == "fail_safe")
            & dense["fallback_used"]
        ][["dataset", "relation_family", "case_id", "active_row_budget"]]
        .drop_duplicates()
        .sort_values(["dataset", "relation_family", "case_id", "active_row_budget"])
        .reset_index(drop=True)
    )
    methods = ("legacy_full_scan_cyclic", "optimized_violated_only_cyclic")
    condition_columns = ["dataset", "relation_family", "case_id", "active_row_budget", "status", "active_sweeps"]
    raw_columns = [
        "dataset", "relation_family", "case_id", "active_row_budget", "fallback_sweeps", "method", "repetition",
        "execution_order", "runtime_ms", "kernel_reported_ms", "sweeps", "projected_rows",
        "verified", "violations_after", "max_residual_after", "error",
    ]

    def part(state: DatasetState) -> dict[str, pd.DataFrame]:
        mine = condition_frame[condition_frame["dataset"] == state.name]
        if mine.empty:
            return {
                "fallback_conditions": pd.DataFrame(columns=condition_columns),
                "fallback_raw": pd.DataFrame(columns=raw_columns),
            }
        cases, _ = dense_cases_for(ctx, PRIMARY_TOLERANCE, state.name)
        case_map = {(case["relation_family"], case["case_id"]): case for case in cases}
        conditions, rows = [], []
        for condition_index, condition in enumerate(mine.itertuples(index=False)):
            case = case_map.get((condition.relation_family, condition.case_id))
            if case is None:
                conditions.append({**condition._asdict(), "status": "case_not_reconstructed", "active_sweeps": -1})
                continue
            snapshot, candidate = case["_snapshot"], case["_candidate"]
            active = core.run_enforcement_method(
                snapshot, candidate, method="arbiter_active", tolerance=PRIMARY_TOLERANCE,
                allow_fallback=False,
                **{**options, "max_sweeps": int(cfg["fallback_condition_sweeps"]),
                   "active_row_budget": int(condition.active_row_budget)},
            )
            if active.verification.passed:
                conditions.append({**condition._asdict(), "status": "no_fallback_needed", "active_sweeps": active.sweeps})
                continue
            conditions.append({**condition._asdict(), "status": "timed", "active_sweeps": active.sweeps})

            def kernel(method: str, sweeps: int):
                core_method = "full_cyclic" if method == methods[0] else "arbiter_active"
                started = time.perf_counter_ns()
                result = core.run_enforcement_method(
                    snapshot, active.prices, method=core_method, tolerance=PRIMARY_TOLERANCE,
                    allow_fallback=False,
                    **{**options, "max_sweeps": int(sweeps), "active_row_budget": snapshot.constraints},
                )
                return result, (time.perf_counter_ns() - started) / 1e6

            for sweeps in cfg["fallback_sweeps_grid"]:
                for warmup in range(int(cfg["fallback_warmups"])):
                    order = list(methods) if (condition_index + warmup) % 2 == 0 else list(reversed(methods))
                    for method in order:
                        kernel(method, int(sweeps))
                for repetition in range(int(cfg["fallback_repetitions"])):
                    ctx.beat("s07 fallback")
                    order = list(methods) if (condition_index + repetition) % 2 == 0 else list(reversed(methods))
                    for execution_order, method in enumerate(order):
                        result, wall_ms = kernel(method, int(sweeps))
                        oracle = core.independent_verify(snapshot, result.prices, tolerance=PRIMARY_TOLERANCE)
                        rows.append(
                            {
                                "dataset": state.name,
                                "relation_family": condition.relation_family,
                                "case_id": condition.case_id,
                                "active_row_budget": int(condition.active_row_budget),
                                "fallback_sweeps": int(sweeps),
                                "method": method,
                                "repetition": int(repetition),
                                "execution_order": int(execution_order),
                                "runtime_ms": float(wall_ms),
                                "kernel_reported_ms": float(result.runtime_ms),
                                "sweeps": int(result.sweeps),
                                "projected_rows": int(result.projected_rows),
                                "verified": bool(oracle.passed),
                                "violations_after": int(oracle.violations),
                                "max_residual_after": float(oracle.max_residual),
                                "error": result.error,
                            }
                        )
                    ctx.beat("s07 fallback")
        return {
            "fallback_conditions": pd.DataFrame(conditions, columns=condition_columns),
            "fallback_raw": pd.DataFrame(rows, columns=raw_columns),
        }

    frames = run_parts(ctx, "s07", [(state.key, lambda state=state: part(state)) for state in ctx.datasets.values()])
    raw = frames.get("fallback_raw", pd.DataFrame(columns=raw_columns))
    conditions = frames.get("fallback_conditions", pd.DataFrame(columns=condition_columns))
    summary_rows, paired_rows = [], []
    if len(raw):
        raw = raw.copy()
        raw["verified"] = as_bool(raw["verified"])
        keys = ["dataset", "relation_family", "case_id", "active_row_budget", "fallback_sweeps"]
        for group_keys, group in raw.groupby(keys + ["method"], sort=True):
            values = group["runtime_ms"].to_numpy(float)
            low, high = bootstrap_median_ci(values, resamples=int(cfg["bootstrap_resamples"]),
                                            seed=stable_seed("fallback", *group_keys))
            summary_rows.append(
                {
                    **dict(zip(keys + ["method"], group_keys)),
                    "repetitions": int(len(group)),
                    "runtime_p50_ms": float(np.median(values)),
                    "runtime_p95_observed_ms": p95_observed(values),
                    "runtime_median_ci95_low_ms": low,
                    "runtime_median_ci95_high_ms": high,
                    "verified_rate": float(group["verified"].mean()),
                    "sweeps_median": float(group["sweeps"].median()),
                    "projected_rows_median": float(group["projected_rows"].median()),
                }
            )
        pivot_time = raw.pivot_table(index=keys + ["repetition"], columns="method", values="runtime_ms", aggfunc="first")
        pivot_ok = raw.pivot_table(index=keys + ["repetition"], columns="method", values="verified", aggfunc="first")
        if set(methods).issubset(pivot_time.columns):
            both_ok = pivot_ok[list(methods)].astype(bool).all(axis=1)
            ratios = (pivot_time[methods[0]] / pivot_time[methods[1]]).rename("ratio")
            table = pd.concat([ratios, both_ok.rename("both_verified")], axis=1).reset_index()
            for group_keys, group in table.groupby(keys, sort=True):
                valid = group.loc[group["both_verified"], "ratio"].to_numpy(float)
                low, high = bootstrap_median_ci(valid, resamples=int(cfg["bootstrap_resamples"]),
                                                seed=stable_seed("fallback-speedup", *group_keys))
                paired_rows.append(
                    {
                        **dict(zip(keys, group_keys)),
                        "paired_repetitions": int(len(group)),
                        "both_verified_repetitions": int(valid.size),
                        "speedup_legacy_over_optimized_p50": float(np.median(valid)) if valid.size else math.nan,
                        "speedup_ci95_low": low,
                        "speedup_ci95_high": high,
                    }
                )
    outputs = {
        "fallback_conditions": conditions,
        "fallback_raw": raw,
        "fallback_summary": pd.DataFrame(summary_rows),
        "fallback_paired": pd.DataFrame(paired_rows),
    }
    write_outputs(ctx, "s07", outputs)
    timed = conditions[conditions["status"] == "timed"] if len(conditions) else conditions
    ctx.add_check("s07", "fallback kernels were exercised at 1e-12", len(timed) > 0, severity="finding",
                  detail=f"{len(timed)} timed conditions")
    if len(raw):
        verified = as_bool(raw["verified"])
        for sweeps in sorted(raw["fallback_sweeps"].unique()):
            mask = raw["fallback_sweeps"] == sweeps
            ctx.add_check(
                "s07", f"both fallback kernels verify within {int(sweeps)} sweeps",
                bool(verified[mask].all()), severity="finding",
                detail=f"verified rate {verified[mask].mean():.3f}",
            )
    return outputs


# ---------------------------------------------------------------------------
# Stage 08: declaration coverage
# ---------------------------------------------------------------------------


def coverage_encodings(relations: pd.DataFrame, cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    families = list(RELATION_FAMILIES)
    encodings = []
    for code in cfg["coverage_family_subsets"]:
        if code == "none":
            chosen: list[str] = []
        else:
            reverse = {value: key for key, value in FAMILY_CODES.items()}
            chosen = [reverse[letter] for letter in code]
        frame = relations[relations["relation_type"].isin(chosen)].reset_index(drop=True)
        encodings.append({"encoding": f"families:{code}", "kind": "family_subset", "fraction": math.nan,
                          "fraction_seed": -1, "relations": frame})
    ordered = relations.reset_index(drop=True)
    for fraction in cfg["coverage_fractions"]:
        for seed in cfg["coverage_fraction_seeds"]:
            keys = ordered["relation_id"].astype(str).map(lambda rid: stable_rank_key("coverage", seed, rid))
            order = np.argsort(keys.to_numpy(), kind="stable")
            keep = int(round(float(fraction) * len(ordered)))
            frame = ordered.iloc[np.sort(order[:keep])].reset_index(drop=True)
            encodings.append({"encoding": f"fraction:{float(fraction):.2f}:seed{int(seed)}", "kind": "random_fraction",
                              "fraction": float(fraction), "fraction_seed": int(seed), "relations": frame})
    _ = families
    return encodings


def run_stage_08(ctx: RevisionContext) -> dict[str, pd.DataFrame]:
    options = ctx.gate_options()

    def part(state: DatasetState) -> dict[str, pd.DataFrame]:
        rows = []
        total = len(state.relations)
        for encoding in coverage_encodings(state.relations, ctx.config):
            encoded = encoding["relations"]
            snapshot = core.build_snapshot(state.asset_ids, encoded, epoch=1)
            counts = encoded["relation_type"].value_counts().to_dict()
            for (model, seed), proposal in state.ordered_proposals():
                partial_before = core.independent_verify(snapshot, proposal, tolerance=PRIMARY_TOLERANCE)
                complete_before = core.independent_verify(state.snapshot, proposal, tolerance=PRIMARY_TOLERANCE)
                result = gate(snapshot, proposal, tolerance=PRIMARY_TOLERANCE, gate_options=options)
                released = not result.rejected
                partial_after = core.independent_verify(snapshot, result.prices, tolerance=PRIMARY_TOLERANCE)
                complete_after = core.independent_verify(state.snapshot, result.prices, tolerance=PRIMARY_TOLERANCE)
                metrics = state.evaluator.metrics(result.prices) if released else {}
                needed = sweeps_needed(ctx, snapshot, proposal, result)
                rows.append(
                    {
                        **needed,
                        "dataset": state.name,
                        "encoding": encoding["encoding"],
                        "encoding_kind": encoding["kind"],
                        "fraction": encoding["fraction"],
                        "fraction_seed": encoding["fraction_seed"],
                        "encoded_relations": int(len(encoded)),
                        "complete_relations": int(total),
                        "encoded_share": float(len(encoded) / total) if total else math.nan,
                        "encoded_decomposition": int(counts.get("decomposition", 0)),
                        "encoded_projection": int(counts.get("projection_derivation", 0)),
                        "encoded_filter": int(counts.get("filter_derivation", 0)),
                        "model": model,
                        "seed": int(seed),
                        "violations_before_encoded": int(partial_before.violations),
                        "violations_before_complete": int(complete_before.violations),
                        "released": released,
                        "fallback_used": bool(result.fallback_used),
                        "sweeps": int(result.sweeps),
                        "gate_ms": float(result.runtime_ms),
                        "encoded_violations_after": int(partial_after.violations),
                        "unsafe_release": bool(released and not partial_after.passed),
                        "complete_violations_after": int(complete_after.violations) if released else -1,
                        "complete_max_residual_after": float(complete_after.max_residual) if released else math.nan,
                        "max_saving_vs_list": metrics.get("max_saving_vs_list", math.nan),
                        "n_saving_gt_1pct": metrics.get("n_saving_gt_1pct", 0),
                        "n_saving_gt_0p1pct": metrics.get("n_saving_gt_0p1pct", 0),
                        "max_saving_conservative": metrics.get("max_saving_conservative", math.nan),
                        "n_conservative_gt_1pct": metrics.get("n_conservative_gt_1pct", 0),
                        "relations_above_1e12": metrics.get("relations_above_1e12", 0),
                        "error": result.error,
                    }
                )
                ctx.beat("s08 coverage")
        return {"coverage_raw": pd.DataFrame(rows)}

    frames = run_parts(ctx, "s08", [(state.key, lambda state=state: part(state)) for state in ctx.datasets.values()])
    raw = frames["coverage_raw"].copy()
    for column in ["released", "fallback_used", "unsafe_release"]:
        raw[column] = as_bool(raw[column])
    released = raw[raw["released"]]
    summary = (
        raw.groupby(["dataset", "encoding", "encoding_kind"], as_index=False)
        .agg(
            encoded_share=("encoded_share", "first"),
            proposals=("model", "size"),
            released_rate=("released", "mean"),
            fallback_rate=("fallback_used", "mean"),
            unsafe_releases=("unsafe_release", "sum"),
            releases_with_complete_violations=("complete_violations_after", lambda values: int((values > 0).sum())),
            complete_violations_after_median=("complete_violations_after", "median"),
            complete_violations_after_max=("complete_violations_after", "max"),
            max_saving_vs_list=("max_saving_vs_list", "max"),
            releases_with_saving_gt_1pct=("n_saving_gt_1pct", lambda values: int((values > 0).sum())),
            max_saving_conservative=("max_saving_conservative", "max"),
            releases_with_conservative_gt_1pct=("n_conservative_gt_1pct", lambda values: int((values > 0).sum())),
            exposed_relations_median=("relations_above_1e12", "median"),
            gate_p50_ms=("gate_ms", "median"),
        )
        .sort_values(["dataset", "encoded_share", "encoding"])
        .reset_index(drop=True)
    )
    outputs = {"coverage_raw": raw, "coverage_summary": summary}
    write_outputs(ctx, "s08", outputs)
    expected = len(ctx.datasets) * len(coverage_encodings(next(iter(ctx.datasets.values())).relations, ctx.config)) \
        * len(MODEL_NAMES) * len(ctx.config["seeds"])
    ctx.add_check("s08", "coverage grid is complete", len(raw) == expected, detail=f"{len(raw)}/{expected}")
    ctx.add_check("s08", "every release passes the encoded snapshot it was verified against",
                  not raw["unsafe_release"].any())
    complete = released[released["encoding"] == "families:DPF"]
    ctx.add_check("s08", "complete encodings leave zero complete-snapshot violations",
                  bool((complete["complete_violations_after"] == 0).all()))
    return outputs


# ---------------------------------------------------------------------------
# Stage 09 and 10: pricing shapes and recompute versus verify
# ---------------------------------------------------------------------------


def policy_components(features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extent and multiplier of arbiter_experiments.raw_controlled_policy."""

    predicate_kind = features["predicate_kind"].to_numpy(str)
    interval = features["interval_width"].to_numpy(float)
    positive_intervals = interval[interval > 0]
    interval_scale = float(positive_intervals.max()) if positive_intervals.size else 1.0
    category_count = features["category_count"].to_numpy(float)
    category_scale = max(float(category_count.max(initial=1.0)), 1.0)
    extent = np.where(
        predicate_kind == "numeric_interval",
        interval / max(interval_scale, 1e-12),
        np.where(predicate_kind == "category_set", category_count / category_scale, 1.0),
    )
    extent = np.maximum(extent, 1e-9)
    output_multiplier = 1.0 + 0.12 * features["output_count"].to_numpy(float)
    aggregate_multiplier = np.where(features["mode"].to_numpy(str) == "additive_aggregate", 1.25, 1.0)
    return extent, output_multiplier * aggregate_multiplier


def tier_levels(extent: np.ndarray, tiers: int) -> tuple[np.ndarray, int]:
    """Equal-count volume tiers; each asset is charged its tier's ceiling extent."""

    values = np.asarray(extent, dtype=np.float64)
    edges = np.unique(np.quantile(values, [k / tiers for k in range(1, tiers)], method="higher"))
    bracket = np.searchsorted(edges, values, side="left")
    levels = np.empty_like(values)
    for index in np.unique(bracket):
        mask = bracket == index
        levels[mask] = values[mask].max()
    return levels, int(np.unique(bracket).size)


def shape_prices(features: pd.DataFrame, shape: str, *, tiers: int = 8) -> tuple[np.ndarray, dict[str, Any]]:
    extent, multiplier = policy_components(features)
    linear = extent * multiplier
    info: dict[str, Any] = {"shape": shape}
    if shape == "linear":
        prices = linear
    elif shape == "concave_sqrt":
        prices = np.sqrt(extent) * multiplier
    elif shape == "tiered_8":
        levels, realized = tier_levels(extent, int(tiers))
        prices = levels * multiplier
        info["realized_tiers"] = realized
    else:
        raise ValueError(f"unknown shape: {shape}")
    if shape != "linear":
        prices = prices * (float(linear.mean()) / float(prices.mean()))
    return prices.astype(np.float64), info


def decomposition_slack(snapshot: core.ConstraintSnapshot, prices: np.ndarray) -> dict[str, float]:
    meta = snapshot.row_metadata
    rows = np.flatnonzero(meta["constraint_type"].astype(str).to_numpy() == "decomposition")
    if rows.size == 0:
        return {"slack_median": math.nan, "slack_p10": math.nan, "tight_share": math.nan}
    matrix = snapshot.matrix
    vector = np.asarray(prices, dtype=np.float64)
    products = matrix.data * vector[matrix.indices]
    sums = np.add.reduceat(products, matrix.indptr[:-1])
    index = {asset_id: position for position, asset_id in enumerate(snapshot.asset_ids)}
    targets = np.asarray([index[t] for t in meta["target_id"].astype(str).to_numpy()[rows]])
    target_price = vector[targets]
    component_sum = target_price - sums[rows]
    relative = (component_sum - target_price) / np.maximum(target_price, 1e-300)
    return {
        "slack_median": float(np.median(relative)),
        "slack_p10": float(np.quantile(relative, 0.10)),
        "tight_share": float(np.mean(relative <= 1e-9)),
    }


def run_stage_09(ctx: RevisionContext) -> dict[str, pd.DataFrame]:
    cfg = ctx.config
    options = ctx.gate_options()

    def part(state: DatasetState, shape: str) -> dict[str, pd.DataFrame]:
        prices, info = shape_prices(state.features, shape, tiers=int(cfg["shape_tiers"]))
        if shape == "linear" and not np.array_equal(prices, state.list_prices):
            raise RuntimeError("linear shape does not reproduce the audited controlled policy")
        feasibility = core.independent_verify(state.snapshot, prices, tolerance=PRIMARY_TOLERANCE)
        if feasibility.passed:
            center, center_source, center_result = prices, "raw_shape_feasible", None
        else:
            center_result = gate(state.snapshot, prices, tolerance=PRIMARY_TOLERANCE, gate_options=options)
            center = center_result.prices if not center_result.rejected else None
            center_source = "gate_repaired_shape" if center is not None else "gate_rejected_shape"
        slack = decomposition_slack(state.snapshot, center) if center is not None else {}
        evaluator = RelationEvaluator(state.snapshot, prices)
        policy_row = {
            "dataset": state.name,
            "shape": shape,
            "realized_tiers": info.get("realized_tiers", math.nan),
            "mean_price": float(prices.mean()),
            "min_price": float(prices.min()),
            "max_price": float(prices.max()),
            "feasible_at_1e12": bool(feasibility.passed),
            "violated_rows": int(feasibility.violations),
            "max_residual": float(feasibility.max_residual),
            "center_source": center_source,
            "center_repair_sweeps": int(center_result.sweeps) if center_result is not None else 0,
            "center_repair_fallback": bool(center_result.fallback_used) if center_result is not None else False,
            "center_l1_adjustment": float(center_result.l1_adjustment) if center_result is not None else 0.0,
            **{f"decomposition_{key}": value for key, value in slack.items()},
        }
        if shape == "linear":
            predictions = ctx.oof_all[(ctx.oof_all["dataset"] == state.name)
                                      & (ctx.oof_all["seed"].isin(cfg["shape_seeds"]))
                                      & (ctx.oof_all["model"].isin(cfg["shape_models"]))].copy()
            pooled = ctx.pooled_metrics_all[(ctx.pooled_metrics_all["dataset"] == state.name)
                                            & (ctx.pooled_metrics_all["seed"].isin(cfg["shape_seeds"]))
                                            & (ctx.pooled_metrics_all["model"].isin(cfg["shape_models"]))].copy()
            pooled["source"] = "audited_v4_oof"
        else:
            targets = pd.DataFrame({"dataset": state.name, "asset_id": state.asset_ids, "target_price": prices})
            _, predictions, pooled = grouped_oof_predictions(
                state.features, targets, seeds=[int(seed) for seed in cfg["shape_seeds"]],
                model_names=list(cfg["shape_models"]), folds=5,
            )
            pooled["source"] = "retrained_on_shape"
        pooled.insert(1, "shape", shape)
        gate_rows = []
        for (model, seed), group in predictions.groupby(["model", "seed"], sort=True):
            mapping = group.set_index(group["asset_id"].astype(str))["oof_prediction"]
            proposal = mapping.loc[state.asset_ids].to_numpy(dtype=np.float64)
            before = core.independent_verify(state.snapshot, proposal, tolerance=PRIMARY_TOLERANCE)
            result = gate(state.snapshot, proposal, tolerance=PRIMARY_TOLERANCE, gate_options=options)
            released = not result.rejected
            oracle = core.independent_verify(state.snapshot, result.prices, tolerance=PRIMARY_TOLERANCE)
            metrics = evaluator.metrics(result.prices) if released else {}
            needed = sweeps_needed(ctx, state.snapshot, proposal, result)
            gate_rows.append(
                {
                    **needed,
                    "dataset": state.name, "shape": shape, "model": str(model), "seed": int(seed),
                    "violations_before": int(before.violations),
                    "released": released, "fallback_used": bool(result.fallback_used),
                    "sweeps": int(result.sweeps), "gate_ms": float(result.runtime_ms),
                    "violations_after": int(oracle.violations),
                    "unsafe_release": bool(released and not oracle.passed),
                    "l1_adjustment": float(result.l1_adjustment),
                    "max_saving_vs_list": metrics.get("max_saving_vs_list", math.nan),
                    "n_saving_gt_1pct": metrics.get("n_saving_gt_1pct", 0),
                    "error": result.error,
                }
            )
            ctx.beat("s09 shapes")
        attack_rows = []
        if center is not None:
            for family in RELATION_FAMILIES:
                family_center, center_policy = relation_family_attack_center(
                    state.snapshot, center, family, tolerance=PRIMARY_TOLERANCE
                )
                attacks = core.exact_box_row_attacks(
                    state.snapshot, family_center, budget_fraction=float(cfg["shape_attack_budget"]),
                    limit=int(cfg["shape_attack_cases"]), tolerance=PRIMARY_TOLERANCE, include_types=[family],
                )
                for rank, attack in enumerate(attacks):
                    result = gate(state.snapshot, attack["candidate"], tolerance=PRIMARY_TOLERANCE, gate_options=options)
                    released = not result.rejected
                    oracle = core.independent_verify(state.snapshot, result.prices, tolerance=PRIMARY_TOLERANCE)
                    attack_rows.append(
                        {
                            "dataset": state.name, "shape": shape, "relation_family": family, "rank": rank,
                            "target_row_id": attack["row_id"], "center_policy": center_policy,
                            "certified_target_residual": float(attack["certified_target_residual"]),
                            "violations_before": int(attack["violations_before"]),
                            "released": released, "fallback_used": bool(result.fallback_used),
                            "sweeps": int(result.sweeps), "gate_ms": float(result.runtime_ms),
                            "violations_after": int(oracle.violations),
                            "unsafe_release": bool(released and not oracle.passed),
                            "requested_cases": int(cfg["shape_attack_cases"]),
                            "available_cases": int(len(attacks)),
                        }
                    )
                ctx.beat("s09 attacks")
        policy_row["attack_cases_available"] = int(len(attack_rows))
        return {
            "shape_policy": pd.DataFrame([policy_row]),
            "shape_model_metrics": pooled.reset_index(drop=True),
            "shape_gate": pd.DataFrame(gate_rows),
            "shape_attacks": pd.DataFrame(attack_rows),
        }

    parts = [
        (f"{state.key}/{shape}", lambda state=state, shape=shape: part(state, shape))
        for state in ctx.datasets.values()
        for shape in cfg["shape_names"]
    ]
    frames = run_parts(ctx, "s09", parts)
    gate_frame = frames["shape_gate"].copy()
    for column in ["released", "fallback_used", "unsafe_release"]:
        gate_frame[column] = as_bool(gate_frame[column])
    attack_frame = frames.get("shape_attacks", pd.DataFrame()).copy()
    if len(attack_frame):
        for column in ["released", "fallback_used", "unsafe_release"]:
            attack_frame[column] = as_bool(attack_frame[column])
    gate_summary = (
        gate_frame.groupby(["dataset", "shape"], as_index=False)
        .agg(
            proposals=("model", "size"),
            violations_before_median=("violations_before", "median"),
            violations_before_max=("violations_before", "max"),
            released_rate=("released", "mean"),
            fallback_rate=("fallback_used", "mean"),
            unsafe_releases=("unsafe_release", "sum"),
            sweeps_max=("sweeps", "max"),
            gate_p50_ms=("gate_ms", "median"),
            max_saving_vs_list=("max_saving_vs_list", "max"),
        )
    )
    attack_summary = (
        attack_frame.groupby(["dataset", "shape"], as_index=False)
        .agg(
            attacks=("rank", "size"),
            attack_success_rate=("violations_before", lambda values: float((values > 0).mean())),
            violations_before_max=("violations_before", "max"),
            released_rate=("released", "mean"),
            unsafe_releases=("unsafe_release", "sum"),
        )
        if len(attack_frame)
        else pd.DataFrame()
    )
    outputs = {
        "shape_policy": frames["shape_policy"],
        "shape_model_metrics": frames["shape_model_metrics"],
        "shape_gate": gate_frame,
        "shape_attacks": attack_frame,
        "shape_gate_summary": gate_summary,
        "shape_attack_summary": attack_summary,
    }
    write_outputs(ctx, "s09", outputs)
    ctx.add_check("s09", "every dataset and shape is evaluated",
                  len(frames["shape_policy"]) == len(ctx.datasets) * len(cfg["shape_names"]))
    ctx.add_check("s09", "no shape-experiment release fails independent verification",
                  not gate_frame["unsafe_release"].any()
                  and not (len(attack_frame) and attack_frame["unsafe_release"].any()))
    return outputs


def run_stage_10(ctx: RevisionContext) -> dict[str, pd.DataFrame]:
    cfg = ctx.config

    def part(state: DatasetState) -> dict[str, pd.DataFrame]:
        rows = []
        for shape in cfg["shape_names"]:
            prices, _ = shape_prices(state.features, shape, tiers=int(cfg["shape_tiers"]))
            verification = core.independent_verify(state.snapshot, prices, tolerance=PRIMARY_TOLERANCE)
            for _ in range(int(cfg["recompute_warmups"])):
                shape_prices(state.features, shape, tiers=int(cfg["shape_tiers"]))
                core.independent_verify(state.snapshot, prices, tolerance=PRIMARY_TOLERANCE)
            for repetition in range(int(cfg["recompute_repetitions"])):
                order = ["recompute", "verify"] if repetition % 2 == 0 else ["verify", "recompute"]
                timings = {}
                for operation in order:
                    started = time.perf_counter_ns()
                    if operation == "recompute":
                        shape_prices(state.features, shape, tiers=int(cfg["shape_tiers"]))
                    else:
                        core.independent_verify(state.snapshot, prices, tolerance=PRIMARY_TOLERANCE)
                    timings[operation] = (time.perf_counter_ns() - started) / 1e6
                rows.append(
                    {
                        "dataset": state.name, "shape": shape, "repetition": repetition,
                        "recompute_ms": timings["recompute"], "verify_ms": timings["verify"],
                        "recomputed_policy_violations": int(verification.violations),
                        "recomputed_policy_max_residual": float(verification.max_residual),
                        "assets": state.snapshot.assets, "constraints": state.snapshot.constraints,
                    }
                )
        return {"recompute_raw": pd.DataFrame(rows)}

    frames = run_parts(ctx, "s10", [(state.key, lambda state=state: part(state)) for state in ctx.datasets.values()])
    raw = frames["recompute_raw"]
    summary = (
        raw.groupby(["dataset", "shape"], as_index=False)
        .agg(
            repetitions=("repetition", "size"),
            recompute_p50_ms=("recompute_ms", "median"),
            recompute_p95_ms=("recompute_ms", q95),
            verify_p50_ms=("verify_ms", "median"),
            verify_p95_ms=("verify_ms", q95),
            recomputed_policy_violations=("recomputed_policy_violations", "first"),
            recomputed_policy_max_residual=("recomputed_policy_max_residual", "first"),
        )
    )
    outputs = {"recompute_raw": raw, "recompute_summary": summary}
    write_outputs(ctx, "s10", outputs)
    linear = summary[summary["shape"] == "linear"]
    ctx.add_check("s10", "the published linear policy has zero violated rows",
                  bool((linear["recomputed_policy_violations"] == 0).all()))
    return outputs


# ---------------------------------------------------------------------------
# Stage 11: buyer history
# ---------------------------------------------------------------------------


def buyer_residual_market(state: DatasetState, history: set[str]) -> dict[str, Any]:
    """The residual market seen by one returning buyer.

    A buyer who owns the answers in ``history`` can reconstruct, at no extra
    cost, every answer whose encoded decomposition parts or derivation source
    are owned or themselves reconstructible. Those answers (``derivable``) and
    the owned answers leave the buyer's price vector: their price for this buyer
    is zero by definition. Every remaining relation is rewritten over the parts
    the buyer still lacks, so a decomposition (t, S) touching the history
    becomes p_t <= sum of p over S minus the owned and derivable answers.

    ``list_caps`` is the alternative policy encoding: instead of residual rows,
    cap p_t by the list prices of the missing parts.
    """

    owned = set(history)
    free = set(owned)
    parsed = []
    for record in state.relations.to_dict(orient="records"):
        if str(record["relation_type"]) == "decomposition":
            components = [str(item) for item in json.loads(str(record["component_ids"]))]
            parsed.append(("decomposition", record, str(record["target_id"]), components))
        else:
            parsed.append(("derivation", record, str(record["derived_id"]), str(record["source_id"])))
    changed = True
    while changed:
        changed = False
        for kind, _, target, parts in parsed:
            if target in free:
                continue
            if (kind == "decomposition" and all(item in free for item in parts)) or (
                kind == "derivation" and parts in free
            ):
                free.add(target)
                changed = True
    list_price = dict(zip(state.asset_ids, state.list_prices))
    base_rows, residual_rows, list_caps = [], [], {}
    for kind, record, target, parts in parsed:
        if target in free:
            continue
        if kind == "derivation":
            base_rows.append(record)
            continue
        missing = [item for item in parts if item not in free]
        if len(missing) == len(parts):
            base_rows.append(record)
            continue
        residual_rows.append(
            {
                "dataset": state.name,
                "relation_id": f"history-residual::{record['relation_id']}",
                "relation_type": "decomposition",
                "target_id": target,
                "component_ids": json.dumps(missing),
                "derived_id": "",
                "source_id": "",
                "sound": True,
                "proof_rule": "buyer_history_residual_decomposition",
            }
        )
        list_caps[target] = min(list_caps.get(target, math.inf), float(sum(list_price[item] for item in missing)))
    remaining = [asset_id for asset_id in state.asset_ids if asset_id not in free]
    columns = list(state.relations.columns)
    return {
        "owned": owned,
        "derivable": sorted(free - owned),
        "remaining": remaining,
        "base": pd.DataFrame(base_rows, columns=columns),
        "residual": pd.DataFrame(residual_rows),
        "list_caps": list_caps,
    }


def run_stage_11(ctx: RevisionContext) -> dict[str, pd.DataFrame]:
    cfg = ctx.config
    options = ctx.gate_options()

    def part(state: DatasetState) -> dict[str, pd.DataFrame]:
        participants = set()
        for record in state.relations.to_dict(orient="records"):
            if str(record["relation_type"]) == "decomposition":
                participants.update(str(item) for item in json.loads(str(record["component_ids"])))
            else:
                participants.add(str(record["source_id"]))
        participants = sorted(participants)
        hgb_seed = min(int(seed) for seed in cfg["seeds"])
        full_proposals = [
            ("list_price", -1, state.list_prices),
            ("hist_gradient_boosting", hgb_seed, state.proposals[("hist_gradient_boosting", hgb_seed)]),
        ]
        position = {asset_id: index for index, asset_id in enumerate(state.asset_ids)}
        rows, diagnostic_rows = [], []
        for size in cfg["history_sizes"]:
            for buyer in range(int(cfg["history_buyers"])):
                ranked = sorted(participants, key=lambda asset: stable_rank_key("history", state.name, size, buyer, asset))
                history = set(ranked[: int(size)])
                market = buyer_residual_market(state, history)
                keep = np.asarray([position[asset_id] for asset_id in market["remaining"]], dtype=np.int64)
                list_v = state.list_prices[keep]
                residual_targets = {str(target) for target in market["residual"].get("target_id", [])}
                for variant in cfg["history_variants"]:
                    started = time.perf_counter_ns()
                    if variant == "residual":
                        relations = (
                            pd.concat([market["base"], market["residual"]], ignore_index=True, sort=False)
                            if len(market["residual"]) else market["base"]
                        )
                        snapshot = core.build_snapshot(market["remaining"], relations, epoch=1)
                    elif variant == "list_cap":
                        snapshot = core.build_snapshot(
                            market["remaining"], market["base"], epoch=1, history_caps=market["list_caps"]
                        )
                    else:
                        raise ValueError(f"unknown history variant: {variant}")
                    build_ms = (time.perf_counter_ns() - started) / 1e6
                    target_mask = np.asarray([asset_id in residual_targets for asset_id in market["remaining"]])
                    for name, seed, full_vector in full_proposals:
                        proposal = full_vector[keep]
                        before = core.independent_verify(snapshot, proposal, tolerance=PRIMARY_TOLERANCE)
                        result = gate(snapshot, proposal, tolerance=PRIMARY_TOLERANCE, gate_options=options)
                        released = not result.rejected
                        oracle = core.independent_verify(snapshot, result.prices, tolerance=PRIMARY_TOLERANCE)
                        delta = result.prices - proposal
                        scale = np.maximum(np.abs(list_v), 1e-300)
                        reduction = -delta[target_mask] / scale[target_mask] if target_mask.any() else np.array([])
                        needed = sweeps_needed(ctx, snapshot, proposal, result)
                        rows.append(
                            {
                                **needed,
                                "dataset": state.name, "history_size": int(size), "buyer": int(buyer),
                                "variant": variant, "proposal": name, "proposal_seed": int(seed),
                                "owned_answers": len(market["owned"]),
                                "derivable_answers": len(market["derivable"]),
                                "remaining_answers": len(market["remaining"]),
                                "residual_rows": int(len(market["residual"])),
                                "list_caps": int(len(market["list_caps"])) if variant == "list_cap" else 0,
                                "snapshot_rows": int(snapshot.constraints),
                                "snapshot_build_ms": float(build_ms),
                                "violations_before": int(before.violations),
                                "released": released, "fallback_used": bool(result.fallback_used),
                                "sweeps": int(result.sweeps), "gate_ms": float(result.runtime_ms),
                                "violations_after": int(oracle.violations),
                                "unsafe_release": bool(released and not oracle.passed),
                                "assets_price_increased": int(np.count_nonzero(delta > 1e-12 * scale)),
                                "assets_price_decreased": int(np.count_nonzero(delta < -1e-12 * scale)),
                                "max_increase_vs_list": float(np.max(delta / scale)) if delta.size else 0.0,
                                "history_target_reduction_median": float(np.median(reduction)) if reduction.size else math.nan,
                                "l1_adjustment": float(result.l1_adjustment),
                                "error": result.error,
                            }
                        )
                    ctx.beat("s11 history")
                # Diagnostic only, never a release path: pinning owned answers
                # to zero with cap rows keeps every asset in the vector, and
                # bounded repair then alternates between cap and relation rows.
                pinned = core.build_snapshot(
                    state.asset_ids, state.relations, epoch=1,
                    history_caps={asset_id: 0.0 for asset_id in market["owned"]},
                )
                for name, seed, full_vector in full_proposals:
                    active = core.run_enforcement_method(
                        pinned, full_vector, method="arbiter_active", tolerance=PRIMARY_TOLERANCE,
                        allow_fallback=False, **options,
                    )
                    diagnostic_rows.append(
                        {
                            "dataset": state.name, "history_size": int(size), "buyer": int(buyer),
                            "encoding": "pinned_zero_caps_all_assets", "proposal": name,
                            "active_repair_verified": bool(active.verification.passed),
                            "sweeps": int(active.sweeps),
                            "max_residual_after": float(active.verification.max_residual),
                            "runtime_ms": float(active.runtime_ms),
                        }
                    )
        return {"history_raw": pd.DataFrame(rows), "history_pinned_diagnostic": pd.DataFrame(diagnostic_rows)}

    frames = run_parts(ctx, "s11", [(state.key, lambda state=state: part(state)) for state in ctx.datasets.values()])
    raw = frames["history_raw"].copy()
    for column in ["released", "fallback_used", "unsafe_release"]:
        raw[column] = as_bool(raw[column])
    diagnostic = frames["history_pinned_diagnostic"].copy()
    diagnostic["active_repair_verified"] = as_bool(diagnostic["active_repair_verified"])
    summary = (
        raw.groupby(["dataset", "variant", "proposal", "history_size"], as_index=False)
        .agg(
            buyers=("buyer", "size"),
            derivable_median=("derivable_answers", "median"),
            remaining_median=("remaining_answers", "median"),
            residual_rows_median=("residual_rows", "median"),
            build_p50_ms=("snapshot_build_ms", "median"),
            build_p95_ms=("snapshot_build_ms", q95),
            violations_before_median=("violations_before", "median"),
            released_rate=("released", "mean"),
            fallback_rate=("fallback_used", "mean"),
            unsafe_releases=("unsafe_release", "sum"),
            sweeps_max=("sweeps", "max"),
            gate_p50_ms=("gate_ms", "median"),
            gate_p95_ms=("gate_ms", q95),
            assets_price_increased_median=("assets_price_increased", "median"),
            history_target_reduction_median=("history_target_reduction_median", "median"),
        )
    )
    diagnostic_summary = (
        diagnostic.groupby(["dataset", "proposal", "history_size"], as_index=False)
        .agg(
            buyers=("buyer", "size"),
            active_repair_verified_rate=("active_repair_verified", "mean"),
            sweeps_median=("sweeps", "median"),
            max_residual_after_median=("max_residual_after", "median"),
        )
    )
    outputs = {
        "history_raw": raw,
        "history_summary": summary,
        "history_pinned_diagnostic": diagnostic,
        "history_pinned_diagnostic_summary": diagnostic_summary,
    }
    write_outputs(ctx, "s11", outputs)
    expected = len(ctx.datasets) * len(cfg["history_sizes"]) * int(cfg["history_buyers"]) \
        * len(cfg["history_variants"]) * 2
    ctx.add_check("s11", "buyer-history grid is complete", len(raw) == expected, detail=f"{len(raw)}/{expected}")
    ctx.add_check("s11", "no buyer-history release fails independent verification", not raw["unsafe_release"].any())
    ctx.add_check("s11", "every buyer-history snapshot admits a verified release", bool(raw["released"].all()),
                  severity="finding", detail=f"{int(raw['released'].sum())}/{len(raw)}")
    ctx.add_check("s11", "buyer-history releases need no fallback", not raw["fallback_used"].any(),
                  severity="finding", detail=f"fallback rate {raw['fallback_used'].mean():.3f}")
    return outputs


# ---------------------------------------------------------------------------
# Stage 12: protocol at 1e-12
# ---------------------------------------------------------------------------


def _stand_in_answer(dataset: str, sql: str) -> list[tuple[str, str]]:
    """Deterministic stand-in payload: the closure fault logic needs answer
    bytes to bind, not query results. DuckDB answers were used in Notebook 05."""

    return [("stand-in-answer", hashlib.sha256(f"{dataset}\x1f{sql}".encode("utf-8")).hexdigest())]


def run_stage_12(ctx: RevisionContext) -> dict[str, pd.DataFrame]:
    cfg = ctx.config
    options = ctx.gate_options()

    def part(state: DatasetState) -> dict[str, pd.DataFrame]:
        feasible = state.list_prices
        attack_v4 = core.exact_box_row_attacks(
            state.snapshot, feasible, budget_fraction=0.50, limit=1, tolerance=AUDITED_TOLERANCE
        )
        attack_new = core.exact_box_row_attacks(
            state.snapshot, feasible, budget_fraction=0.50, limit=1, tolerance=PRIMARY_TOLERANCE
        )
        violating = attack_v4[0]["candidate"]
        repair_v4 = core.run_enforcement_method(
            state.snapshot, violating, method="arbiter_active", tolerance=AUDITED_TOLERANCE, **options
        )
        repair_new = core.run_enforcement_method(
            state.snapshot, violating, method="arbiter_active", tolerance=PRIMARY_TOLERANCE, **options
        )
        policy_check = core.independent_verify(state.snapshot, feasible, tolerance=PRIMARY_TOLERANCE)
        v4_at_new = core.independent_verify(state.snapshot, repair_v4.prices, tolerance=PRIMARY_TOLERANCE)
        vectors = pd.DataFrame(
            [
                {
                    "dataset": state.name,
                    "controlled_policy_verifies_at_1e12": bool(policy_check.passed),
                    "controlled_policy_max_residual": float(policy_check.max_residual),
                    "notebook02_attack_row_same_at_1e12": bool(attack_v4[0]["row_id"] == attack_new[0]["row_id"]),
                    "notebook02_candidate_same_at_1e12": bool(
                        np.array_equal(attack_v4[0]["candidate"], attack_new[0]["candidate"])
                    ),
                    "repair_at_1e7_verifies_at_1e12": bool(v4_at_new.passed),
                    "repair_at_1e7_max_residual": float(v4_at_new.max_residual),
                    "repair_at_1e12_verified": bool(repair_new.verification.passed),
                    "repair_vectors_identical": bool(np.array_equal(repair_v4.prices, repair_new.prices)),
                    "repair_max_abs_difference": float(np.max(np.abs(repair_v4.prices - repair_new.prices))),
                    "repair_sweeps_1e7": int(repair_v4.sweeps),
                    "repair_sweeps_1e12": int(repair_new.sweeps),
                    "repair_ms_1e7": float(repair_v4.runtime_ms),
                    "repair_ms_1e12": float(repair_new.runtime_ms),
                }
            ]
        )
        faults = run_fault_injection_campaign(
            snapshot=state.snapshot, feasible_prices=feasible, violating_prices=violating,
            tolerance=PRIMARY_TOLERANCE,
        )
        faults.insert(0, "dataset", state.name)
        concurrency = run_stale_epoch_concurrency_campaign(
            snapshot=state.snapshot, feasible_prices=feasible,
            trials=int(cfg["protocol_concurrency_trials"]), tolerance=PRIMARY_TOLERANCE,
        )
        concurrency.insert(0, "dataset", state.name)
        bundle = ctx.bundles(PRIMARY_TOLERANCE)[state.name]
        with tempfile.TemporaryDirectory(prefix="arbiter-revision-protocol-") as work:
            closure_faults, closure_concurrency, delivery, _ = closure.run_protocol_fault_campaign(
                {state.name: bundle}, answer_executor=_stand_in_answer, work_dir=Path(work),
                tolerance=PRIMARY_TOLERANCE, concurrency_trials=int(cfg["closure_concurrency_trials"]),
            )
        ctx.beat("s12 protocol")
        return {
            "protocol_vectors": vectors,
            "v4_faults": faults,
            "v4_concurrency": concurrency,
            "closure_faults": closure_faults,
            "closure_concurrency": closure_concurrency,
            "closure_delivery": delivery,
        }

    frames = run_parts(ctx, "s12", [(state.key, lambda state=state: part(state)) for state in ctx.datasets.values()])
    summary_rows = []
    for name in ["v4_faults", "v4_concurrency", "closure_faults", "closure_concurrency", "closure_delivery"]:
        frame = frames[name]
        passed = as_bool(frame["passed"])
        summary_rows.append({"campaign": name, "rows": int(len(frame)), "passed": int(passed.sum()),
                             "pass_rate": float(passed.mean()) if len(frame) else math.nan})
    outputs = {**frames, "protocol_summary": pd.DataFrame(summary_rows)}
    write_outputs(ctx, "s12", outputs)
    vectors = frames["protocol_vectors"]
    ctx.add_check("s12", "protocol vectors verify at 1e-12",
                  bool(as_bool(vectors["controlled_policy_verifies_at_1e12"]).all()
                       and as_bool(vectors["repair_at_1e12_verified"]).all()))
    for name in ["v4_faults", "v4_concurrency", "closure_faults", "closure_concurrency", "closure_delivery"]:
        ctx.add_check("s12", f"{name} behave as specified at 1e-12", bool(as_bool(frames[name]["passed"]).all()),
                      detail=f"{int(as_bool(frames[name]['passed']).sum())}/{len(frames[name])}")
    return outputs


STAGE_FUNCTIONS: dict[str, Callable[[RevisionContext], dict[str, pd.DataFrame]]] = {
    "s01": run_stage_01,
    "s02": run_stage_02,
    "s03": run_stage_03,
    "s04": run_stage_04,
    "s05": run_stage_05,
    "s06": run_stage_06,
    "s07": run_stage_07,
    "s08": run_stage_08,
    "s09": run_stage_09,
    "s10": run_stage_10,
    "s11": run_stage_11,
    "s12": run_stage_12,
}

STAGE_HEADLINES: dict[str, str] = {
    "s01": "tolerance_overall",
    "s02": "attack_summary",
    "s03": "gap_summary",
    "s04": "scale_summary",
    "s05": "ablation_summary",
    "s06": "dense_budget_overview",
    "s07": "fallback_paired",
    "s08": "coverage_summary",
    "s09": "shape_gate_summary",
    "s10": "recompute_summary",
    "s11": "history_summary",
    "s12": "protocol_summary",
}


def run_stage(ctx: RevisionContext, stage: str) -> pd.DataFrame:
    """Run (or resume) one stage and return its headline table."""

    if stage not in STAGE_FUNCTIONS:
        raise ValueError(f"unknown stage {stage}")
    ctx.log(f"Stage {stage} started: {STAGES[stage]}")
    started = time.perf_counter()
    outputs = STAGE_FUNCTIONS[stage](ctx)
    ctx.stage_wall_s[stage] = time.perf_counter() - started
    failures = [
        row for row in ctx.quality
        if row["stage"] == stage and row["severity"] == "fatal" and not row["passed"]
    ]
    ctx.log(f"Stage {stage} finished in {ctx.stage_wall_s[stage] / 60:.1f} min")
    if failures:
        raise RuntimeError(
            f"Stage {stage} failed an integrity check: "
            + "; ".join(row["check"] for row in failures)
            + ". Send the traceback and this notebook's output to the auditor."
        )
    headline = outputs.get(STAGE_HEADLINES[stage], pd.DataFrame())
    return headline


def run_stage_guarded(ctx: RevisionContext, stage: str) -> pd.DataFrame | None:
    """Run one stage; on an exception, record it and let later stages run.

    One run then surfaces every problem at once, and finished stages stay
    checkpointed. The final cell reports all failures and refuses to call the
    run a PASS.
    """

    try:
        return run_stage(ctx, stage)
    except Exception as exc:  # noqa: BLE001 - recorded and re-raised at the end
        import traceback

        text = traceback.format_exc()
        atomic_write_text(ctx.output_dir / "errors" / f"{stage}.txt", text)
        ctx.add_check(
            stage,
            f"stage {stage} completed without an exception",
            False,
            detail=f"{type(exc).__name__}: {exc}",
        )
        ctx.log(
            f"Stage {stage} FAILED: {type(exc).__name__}: {exc}. Later stages still "
            "run; the final cell lists every failure."
        )
        return None


# ---------------------------------------------------------------------------
# Runtime estimate (pilot)
# ---------------------------------------------------------------------------


def _interpolate_cost(scales: Sequence[int], costs: Sequence[float], target: int) -> float:
    """Log-log interpolation of a per-call cost; linear extrapolation outside."""

    points = sorted(
        (int(s), float(c)) for s, c in zip(scales, costs) if math.isfinite(float(c)) and float(c) > 0
    )
    if not points:
        return math.nan
    if len(points) == 1:
        return points[0][1] * (target / points[0][0])
    xs = np.log([point[0] for point in points])
    ys = np.log([point[1] for point in points])
    x = math.log(target)
    if xs[0] <= x <= xs[-1]:
        return float(np.exp(np.interp(x, xs, ys)))
    if x > xs[-1]:
        slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        return float(np.exp(ys[-1] + slope * (x - xs[-1])))
    slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
    return float(np.exp(ys[0] + slope * (x - xs[0])))


def estimate_full_runtime(ctx: RevisionContext) -> pd.DataFrame:
    """Extrapolate the full-mode runtime from what the pilot measured."""

    pilot, full = ctx.config, mode_config("full")
    provenance = ctx.checkpointer.provenance()
    elapsed = (
        provenance[provenance["valid_for_this_run"].astype(bool)]
        .groupby("stage")["elapsed_s"].sum().to_dict()
        if len(provenance) else {}
    )
    rows = []

    def add(stage: str, estimate_s: float, basis: str) -> None:
        rows.append({"stage": stage, "description": STAGES.get(stage, stage),
                     "pilot_seconds": float(elapsed.get(stage, math.nan)),
                     "estimated_full_seconds": float(estimate_s), "basis": basis})

    datasets = len(ctx.datasets)
    proposals_pilot = datasets * len(MODEL_NAMES) * len(pilot["seeds"])
    proposals_full = datasets * len(MODEL_NAMES) * len(full["seeds"])
    outputs = ctx.stage_outputs
    # s01
    natural_s = elapsed.get("s01", 0.0)
    heavy = outputs.get("s01", {}).get("heavy_raw", pd.DataFrame())
    if len(heavy):
        per_call = heavy.groupby("method")["runtime_ms"].mean().sum() / 1000.0
        heavy_full = per_call * proposals_full * (full["heavy_repetitions"] + full["heavy_warmups"])
        natural_part = outputs["s01"]["natural_gate"]["gate_ms"].sum() / 1000.0
        add("s01", natural_part * proposals_full / proposals_pilot + heavy_full * 1.15,
            "natural gates scaled by proposal count; heavy-load mean cost per method times full calls")
    else:
        add("s01", natural_s * proposals_full / max(proposals_pilot, 1), "pilot time scaled by proposal count")
    # s02 and s03
    cells_pilot = datasets * len(pilot["attack_budgets"]) * 3
    cells_full = datasets * len(full["attack_budgets"]) * 3
    for stage, cases_key in [("s02", "attack_cases"), ("s03", "gap_cases")]:
        pilot_s = elapsed.get(stage, math.nan)
        add(stage, pilot_s * (cells_full / cells_pilot) * max(1.0, (full[cases_key] / pilot[cases_key]) ** 0.5),
            "pilot time times cell ratio, with a square-root factor for more cases per cell")
    # s04
    scale_raw = outputs.get("s04", {}).get("scale_raw", pd.DataFrame())
    builds = outputs.get("s04", {}).get("build_raw", pd.DataFrame())
    if len(scale_raw):
        total = 0.0
        cost = scale_raw.groupby(["method", "scale"])["runtime_ms"].mean().reset_index()
        build_cost = builds.groupby("scale")["snapshot_build_ms"].mean() if len(builds) else pd.Series(dtype=float)
        cases_full = 3 * full["scale_cases_per_family"]
        osqp_rows = cost[cost["method"] == "osqp_qp"]
        osqp_runs = full["scale_repetitions"] + full["scale_warmups"]
        if len(osqp_rows):
            largest_ms = float(osqp_rows.loc[osqp_rows["scale"].idxmax(), "runtime_ms"])
            for limit, repetitions, warmups in OSQP_REPETITION_RULE:
                if largest_ms <= limit:
                    osqp_runs = repetitions + warmups
                    break
        for scale in full["scale_points"]:
            for method in full["scale_methods"]:
                sub = cost[cost["method"] == method]
                per_call_ms = _interpolate_cost(sub["scale"].tolist(), sub["runtime_ms"].tolist(), int(scale))
                runs = osqp_runs if method == "osqp_qp" else full["scale_repetitions"] + full["scale_warmups"]
                total += datasets * cases_full * runs * per_call_ms / 1000.0
            if len(build_cost):
                build_ms = _interpolate_cost(build_cost.index.tolist(), build_cost.tolist(), int(scale))
                total += datasets * full["scale_repetitions"] * build_ms / 1000.0
        add("s04", total * 1.10, "per-method cost interpolated in log-log between measured scales")
    else:
        add("s04", math.nan, "no pilot data")
    # s05
    ablation = outputs.get("s05", {}).get("ablation_raw", pd.DataFrame())
    if len(ablation):
        per_call = ablation["runtime_ms"].mean() / 1000.0
        calls = datasets * len(ABLATION_SPECS) * 3 * full["scale_cases_per_family"] * (full["ablation_repetitions"] + full["ablation_warmups"])
        add("s05", per_call * calls * 1.2 + elapsed.get("s05", 0.0), "mean gate cost times full calls plus setup")
    else:
        add("s05", elapsed.get("s05", math.nan), "pilot time")
    # s06
    factor = (len(full["dense_budgets"]) * (full["dense_repetitions"] + full["dense_warmups"] + 0.5)) / (
        len(pilot["dense_budgets"]) * (pilot["dense_repetitions"] + pilot["dense_warmups"] + 0.5))
    add("s06", elapsed.get("s06", math.nan) * factor, "pilot time times budget and repetition ratio")
    # s07
    fallback = outputs.get("s07", {}).get("fallback_raw", pd.DataFrame())
    if len(fallback):
        per_pair = fallback.groupby("method")["runtime_ms"].mean().sum() / 1000.0
        conditions = fallback[["dataset", "case_id", "active_row_budget"]].drop_duplicates().shape[0]
        conditions_full = conditions * len(full["dense_budgets"]) / len(pilot["dense_budgets"])
        add("s07", per_pair * conditions_full * (full["fallback_repetitions"] + full["fallback_warmups"]),
            "mean kernel cost times conditions and repetitions")
    else:
        add("s07", 60.0, "no fallback condition in the pilot; nominal allowance")
    # s08
    encodings_pilot = len(pilot["coverage_family_subsets"]) + len(pilot["coverage_fractions"]) * len(pilot["coverage_fraction_seeds"])
    encodings_full = len(full["coverage_family_subsets"]) + len(full["coverage_fractions"]) * len(full["coverage_fraction_seeds"])
    add("s08", elapsed.get("s08", math.nan) * (encodings_full / encodings_pilot) * (proposals_full / proposals_pilot),
        "pilot time times encoding and proposal ratio")
    # s09
    add("s09", elapsed.get("s09", math.nan) * len(full["shape_seeds"]) / len(pilot["shape_seeds"]) * 1.1,
        "pilot time times seed ratio")
    add("s10", elapsed.get("s10", math.nan) * (full["recompute_repetitions"] + full["recompute_warmups"])
        / (pilot["recompute_repetitions"] + pilot["recompute_warmups"]), "pilot time times repetition ratio")
    add("s11", elapsed.get("s11", math.nan) * (len(full["history_sizes"]) * full["history_buyers"])
        / (len(pilot["history_sizes"]) * pilot["history_buyers"]) * 1.3, "pilot time times buyer ratio")
    add("s12", elapsed.get("s12", math.nan) * 3.0 + 60.0, "pilot time with more concurrency trials")
    table = pd.DataFrame(rows)
    table["estimated_full_minutes"] = table["estimated_full_seconds"] / 60.0
    return table


# ---------------------------------------------------------------------------
# Final audit and package
# ---------------------------------------------------------------------------


def _value(frame: pd.DataFrame, column: str, default: Any = math.nan) -> Any:
    if frame is None or not len(frame) or column not in frame:
        return default
    return frame[column].iloc[0]


SWEEP_BUDGETS = (80, 160, 400, 1000)


def sweep_budget_curve(ctx: RevisionContext) -> pd.DataFrame:
    """Share of gate calls that need more sweeps than a given budget.

    Built from the sweeps_to_verify column, which records what active-row
    repair alone needs (fallback disabled) for every gate call of the listed
    experiment families.
    """

    out = ctx.stage_outputs
    families = []
    natural = out.get("s01", {}).get("natural_gate")
    if natural is not None and len(natural):
        for tolerance, group in natural.groupby("tolerance"):
            families.append((f"s01 natural proposals, tolerance {tolerance:.0e}", group))
    for stage, name, label, split in [
        ("s08", "coverage_raw", "s08 declaration coverage", None),
        ("s09", "shape_gate", "s09 pricing-shape proposals", "shape"),
        ("s11", "history_raw", "s11 buyer history", "variant"),
    ]:
        frame = out.get(stage, {}).get(name)
        if frame is None or not len(frame):
            continue
        if split is None:
            families.append((label, frame))
        else:
            for key, group in frame.groupby(split):
                families.append((f"{label}, {key}", group))
    rows = []
    cap = int(ctx.config["sweeps_to_verify_cap"])
    for label, frame in families:
        sweeps = frame["sweeps_to_verify"].to_numpy(dtype=float)
        within = as_bool(frame["verified_within_cap"]).to_numpy()
        row = {
            "family": label,
            "gate_calls": int(len(frame)),
            "sweeps_p50": float(np.median(sweeps)) if sweeps.size else math.nan,
            "sweeps_p95": float(np.quantile(sweeps, 0.95)) if sweeps.size else math.nan,
            "sweeps_max": float(sweeps.max()) if sweeps.size else math.nan,
            "not_verified_within_cap": int((~within).sum()),
            "cap": cap,
        }
        for budget in SWEEP_BUDGETS:
            row[f"share_needing_more_than_{budget}"] = float(np.mean((sweeps > budget) | ~within)) if sweeps.size else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def build_claim_audit(ctx: RevisionContext) -> pd.DataFrame:
    out = ctx.stage_outputs
    claims = []

    def claim(text: str, status: str, value: str, wording: str) -> None:
        claims.append({"claim": text, "status": status, "value": value, "required_wording": wording})

    unsafe_total = 0
    for stage, frames in out.items():
        for name, frame in frames.items():
            if isinstance(frame, pd.DataFrame) and "unsafe_release" in frame.columns and len(frame):
                unsafe_total += int(as_bool(frame["unsafe_release"]).sum())
    claim(
        "No measured release failed independent verification at the stated tolerance.",
        "SUPPORTED" if unsafe_total == 0 else "NOT_SUPPORTED",
        f"unsafe releases across all raw tables = {unsafe_total}",
        "State the empirical count, the tolerance 1e-12, and the encoded scope.",
    )
    natural = out.get("s01", {}).get("natural_gate")
    if natural is not None and len(natural):
        frame = natural.copy()
        frame["released"] = as_bool(frame["released"])
        frame["fallback_used"] = as_bool(frame["fallback_used"])
        primary = frame[frame["tolerance"] == PRIMARY_TOLERANCE]
        audited = frame[frame["tolerance"] == AUDITED_TOLERANCE]
        claim(
            "Every natural proposal is released at 1e-12 by active-row repair without fallback.",
            "SUPPORTED" if bool(primary["released"].all() and not primary["fallback_used"].any()) else "LIMITED",
            f"released {int(primary['released'].sum())}/{len(primary)}; fallback {int(primary['fallback_used'].sum())}; "
            f"max sweeps {int(primary['sweeps'].max())} of the {int(ctx.config['repair_sweeps'])}-sweep budget",
            "Report the sweep maximum against the sweep budget.",
        )
        gt1_audited = int((audited["n_saving_gt_1pct"] > 0).sum())
        gt01_primary = int((primary["n_saving_gt_0p1pct"] > 0).sum())
        claim(
            "At 1e-7 some released vectors still leave an encoded relation with a saving above 1% of the list price; at 1e-12 none exceeds 0.1%.",
            "SUPPORTED" if gt1_audited > 0 and gt01_primary == 0 else "LIMITED",
            f"1e-7: {gt1_audited}/{len(audited)} releases with a saving above 1% (max {audited['max_saving_vs_list'].max():.4f}); "
            f"1e-12: {gt01_primary}/{len(primary)} above 0.1% (max {primary['max_saving_vs_list'].max():.2e})",
            "Savings are relative to the seller list price of the target answer.",
        )
    attacks = out.get("s02", {}).get("attacks")
    if attacks is not None and len(attacks):
        comparison = out["s02"]["attack_v4_comparison"]
        claim(
            "Every exact per-row box attack is repaired and released at 1e-12.",
            "SUPPORTED" if as_bool(attacks["released"]).all() else "LIMITED",
            f"released {int(as_bool(attacks['released']).sum())}/{len(attacks)}; attacks on the same target rows as v4 "
            f"{int(comparison['same_target_rows'].sum())}/{int(comparison['cases_1e12'].sum())}; same release decision "
            f"{int(comparison['same_release_decision'].sum())}/{int(comparison['same_target_rows'].sum())}",
            "Call it an exact per-row box attack, not a global worst-case adversary.",
        )
    scale = out.get("s04", {}).get("scale_summary")
    if scale is not None and len(scale):
        osqp = scale[scale["method"] == "osqp_qp"]
        stop = scale[scale["method"] == "osqp_qp_v4_stop"]
        claim(
            "OSQP configured at 1e-12 certifies the scale candidates at 1e-12.",
            "SUPPORTED" if bool((osqp["verified_rate"] == 1.0).all()) else "LIMITED",
            f"verified rate min {osqp['verified_rate'].min():.3f}; with the audited stopping rule min {stop['verified_rate'].min():.3f}",
            "Report OSQP certification rates next to its latency; a rejected OSQP output is not a release.",
        )
    coverage = out.get("s08", {}).get("coverage_summary")
    if coverage is not None and len(coverage):
        partial = coverage[coverage["encoding"] != "families:DPF"]
        claim(
            "Releases verified against a partial declaration can leave violations of the complete snapshot.",
            "SUPPORTED" if bool((partial["releases_with_complete_violations"] > 0).any()) else "NOT_SUPPORTED",
            f"encodings with surviving complete-snapshot violations: "
            f"{int((partial['releases_with_complete_violations'] > 0).sum())}/{len(partial)}",
            "Describe only the measured encodings; do not extrapolate to arbitrary SQL.",
        )
    shapes = out.get("s09", {}).get("shape_policy")
    if shapes is not None and len(shapes):
        infeasible = shapes[~as_bool(shapes["feasible_at_1e12"])]
        claim(
            "The release decision does not depend on the seller policy being feasible by construction.",
            "SUPPORTED" if len(infeasible) > 0 else "LIMITED",
            f"infeasible shape policies: {len(infeasible)} of {len(shapes)} dataset-shape cells",
            "Name the three shapes and report the violated-row counts of each.",
        )
    history = out.get("s11", {}).get("history_raw")
    if history is not None and len(history):
        frame = history.copy()
        frame["released"] = as_bool(frame["released"])
        frame["fallback_used"] = as_bool(frame["fallback_used"])
        claim(
            "Buyer-specific residual snapshots release by active-row repair without fallback.",
            "SUPPORTED" if bool(frame["released"].all() and not frame["fallback_used"].any()) else "LIMITED",
            f"released {int(frame['released'].sum())}/{len(frame)}; fallback {int(frame['fallback_used'].sum())}; "
            f"max sweeps needed {int(frame['sweeps_to_verify'].max())}",
            "Owned and derivable answers leave the buyer's vector; state the history sizes tested.",
        )
    protocol = out.get("s12", {}).get("protocol_summary")
    if protocol is not None and len(protocol):
        ok = bool((protocol["pass_rate"] == 1.0).all())
        claim(
            "The release protocol behaves identically at 1e-12 (faults, stale epochs, request binding, races).",
            "SUPPORTED" if ok else "NOT_SUPPORTED",
            "; ".join(f"{row.campaign} {row.passed}/{row.rows}" for row in protocol.itertuples(index=False)),
            "Closure payloads in this campaign are deterministic stand-ins; Notebook 05 used DuckDB answers.",
        )
    provenance = ctx.checkpointer.provenance()
    timing_stages = {"s01", "s04", "s05", "s06", "s07", "s10", "s11"}
    hosts = provenance[provenance["stage"].isin(timing_stages)]["host_digest"].nunique() if len(provenance) else 0
    claim(
        "All timing parts ran in one runtime session on one machine.",
        "SUPPORTED" if hosts <= 1 else "LIMITED",
        f"distinct runtime hosts across timing parts: {hosts}",
        "If LIMITED, report that timings span resumed sessions and list the CPU models.",
    )
    osqp_version = _package_version("osqp")
    claim(
        "The OSQP baseline uses the audited OSQP version 1.0.4.",
        "SUPPORTED" if osqp_version == "1.0.4" else "LIMITED",
        f"osqp {osqp_version}",
        "Report the OSQP version used for every OSQP number.",
    )
    claim(
        "Performance generalizes across hardware environments.",
        "NOT_SUPPORTED",
        "one environment",
        "Report the CPU model and core count; do not generalize latency.",
    )
    claim(
        "Buyer-history rows give a sequence-level arbitrage guarantee.",
        "NOT_SUPPORTED",
        "one returning buyer per snapshot",
        "Residual rows remove encoded savings for one buyer's history at release time only.",
    )
    return pd.DataFrame(claims)


def headline_numbers(ctx: RevisionContext) -> dict[str, Any]:
    out = ctx.stage_outputs
    numbers: dict[str, Any] = {"revision_suite_version": REVISION_SUITE_VERSION, "mode": ctx.mode}
    natural = out.get("s01", {}).get("tolerance_overall")
    if natural is not None and len(natural):
        numbers["tolerance_overall"] = natural.to_dict(orient="records")
    scale = out.get("s04", {}).get("scale_summary")
    if scale is not None and len(scale):
        top = scale[scale["scale"] == scale["scale"].max()]
        numbers["scale_largest"] = top[["dataset", "method", "runtime_p50_ms", "verified_rate"]].to_dict(orient="records")
    protocol = out.get("s12", {}).get("protocol_summary")
    if protocol is not None and len(protocol):
        numbers["protocol"] = protocol.to_dict(orient="records")
    return numbers


def finalize_campaign(
    ctx: RevisionContext,
    *,
    embedded_sources: Mapping[str, str],
    embedded_hashes: Mapping[str, str],
    notebook_name: str,
    environment_label: str,
) -> dict[str, Any]:
    """Write audit files, verify the input is unchanged, and build the ZIP."""

    output_dir = ctx.output_dir
    package_state_after = snapshot_folder_state(ctx.package_dir)
    unchanged = package_state_after == ctx.package_state_before
    ctx.add_check("final", "audited-v4 package is byte-identical after the campaign", unchanged)
    missing = [stage for stage in STAGE_ORDER if stage not in ctx.stage_outputs]
    ctx.add_check("final", "every stage produced outputs", not missing, detail=", ".join(missing))
    quality = pd.DataFrame(ctx.quality)
    atomic_write_csv(output_dir / "stage_quality_checks.csv", quality)
    claims = build_claim_audit(ctx)
    atomic_write_csv(output_dir / "claim_audit.csv", claims)
    try:
        curve = sweep_budget_curve(ctx)
    except Exception as exc:  # noqa: BLE001 - a derived table must never block packaging
        curve = pd.DataFrame([{"family": f"curve failed: {type(exc).__name__}: {exc}"}])
    atomic_write_csv(ctx.outputs_dir / "final_sweep_budget_curve.csv", curve)
    atomic_write_csv(output_dir / "input_audit.csv", ctx.input_audit)
    atomic_write_csv(output_dir / "part_provenance.csv", ctx.checkpointer.provenance())
    if ctx.mode == "pilot":
        try:
            estimate = estimate_full_runtime(ctx)
        except Exception as exc:  # noqa: BLE001 - an estimate must never block packaging
            estimate = pd.DataFrame(
                [{"stage": "all", "description": f"estimate failed: {type(exc).__name__}: {exc}",
                  "pilot_seconds": math.nan, "estimated_full_seconds": math.inf, "basis": "error",
                  "estimated_full_minutes": math.inf}]
            )
        atomic_write_csv(output_dir / "runtime_estimate.csv", estimate)
    atomic_write_text(output_dir / "headline_numbers.json",
                      json.dumps(headline_numbers(ctx), indent=2, sort_keys=True, default=str))
    atomic_write_text(output_dir / "revision_run_config.json", json.dumps(ctx.config, indent=2, sort_keys=True))
    atomic_write_text(
        output_dir / "environment.json",
        json.dumps(
            {
                **runtime_fingerprint(ctx.session_id),
                "audited_input": {
                    "package_dir_name": ctx.package_dir.name,
                    "manifest_sha256": AUDITED_MANIFEST_SHA256,
                    "run_config_sha256": AUDITED_RUN_CONFIG_SHA256,
                },
                "notebook": notebook_name,
                "environment_label": environment_label,
                "stage_wall_seconds": ctx.stage_wall_s,
            },
            indent=2,
            sort_keys=True,
        ),
    )
    code_dir = output_dir / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    for name, text in embedded_sources.items():
        atomic_write_text(code_dir / name, text)
    atomic_write_text(code_dir / "code_manifest.json", json.dumps(dict(embedded_hashes), indent=2, sort_keys=True))
    fatal_failed = quality[(quality["severity"] == "fatal") & (~quality["passed"].astype(bool))]
    status = "PASS" if fatal_failed.empty else "FAILED"
    readme = "\n".join(
        [
            f"# ARBITER revision campaign ({ctx.mode}, {environment_label})",
            "",
            f"Suite version: {REVISION_SUITE_VERSION}. Status: {status}.",
            "",
            "This package re-executes every tolerance-dependent experiment of the audited-v4",
            "campaign at verifier tolerance 1e-12 and adds four replacement experiments.",
            "The audited-v4 package was read-only input and is byte-identical after the run",
            f"({'confirmed' if unchanged else 'NOT confirmed'}).",
            "",
            "Prices, attacks, and buyer histories are controlled deterministic constructions,",
            "not market observations. TPC-H is benchmark-generated. Read claim_audit.csv",
            "before drafting manuscript text; LIMITED and NOT_SUPPORTED rows remain visible.",
            "Raw tables under outputs/ are the numerical authority.",
            "",
        ]
    )
    atomic_write_text(output_dir / "README.md", readme)
    rows = []
    for path in sorted(output_dir.rglob("*")):
        relative = path.relative_to(output_dir)
        if not path.is_file() or relative.parts[0] == "parts" or path.name in {"checksums.csv", "checksums.sha256", "run.lock"}:
            continue
        if path.name.startswith("."):
            continue
        rows.append({"path": relative.as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = pd.DataFrame(rows)
    atomic_write_csv(output_dir / "checksums.csv", manifest)
    atomic_write_text(output_dir / "checksums.sha256",
                      "".join(f"{row.sha256}  {row.path}\n" for row in manifest.itertuples(index=False)))
    label = slug(environment_label)
    suffix = "" if status == "PASS" else "_FAILED"
    zip_path = output_dir.parent / f"arbiter_revision_{ctx.mode}_{label}_v1_0{suffix}.zip"
    for stale in output_dir.parent.glob(f"arbiter_revision_{ctx.mode}_{label}_v1_0*.zip"):
        stale.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for row in manifest.itertuples(index=False):
            archive.write(output_dir / row.path, f"{output_dir.name}/{row.path}")
        for name in ["checksums.csv", "checksums.sha256"]:
            archive.write(output_dir / name, f"{output_dir.name}/{name}")
    with zipfile.ZipFile(zip_path) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("revision ZIP failed its integrity test")
        for row in manifest.itertuples(index=False):
            if sha256_bytes(archive.read(f"{output_dir.name}/{row.path}")) != row.sha256:
                raise RuntimeError(f"revision ZIP checksum mismatch: {row.path}")
    return {
        "status": status,
        "zip_path": zip_path,
        "zip_bytes": zip_path.stat().st_size,
        "fatal_failures": fatal_failed["check"].tolist(),
        "quality": quality,
        "claims": claims,
    }
