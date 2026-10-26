"""Core implementation used by the Arbiter ICDE Colab evidence suite.

The module deliberately separates:

* proposal generation (untrusted and replaceable),
* sparse repair (a liveness optimization),
* an independent verifier (the release authority), and
* an epoch-bound compare-and-commit gate.

The verifier certifies only the linear constraints encoded in a snapshot.  It
does not claim universal SQL derivability or economic optimality.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import hmac
import json
import math
import secrets
import threading
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import LinearConstraint, minimize

try:  # Colab installs OSQP; the SciPy fallback keeps local smoke tests usable.
    import osqp  # type: ignore
except Exception:  # pragma: no cover - exercised only where OSQP is unavailable.
    osqp = None


DEFAULT_TOLERANCE = 1e-7


def _validated_tolerance(value: float) -> float:
    """Return a finite, strictly positive verifier tolerance.

    A NaN tolerance makes every ``residual > tolerance`` comparison false, so
    accepting caller-controlled non-finite tolerances would be a fail-open bug.
    """

    tolerance = float(value)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and strictly positive")
    return tolerance


def _strict_sound_flag(value: Any) -> bool:
    """Parse a relation soundness flag without Python truthiness surprises."""

    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    raise ValueError("relation sound must be an explicit boolean true/false")


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_hex(*parts: bytes) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part)
    return digest.hexdigest()


def asset_order_digest(asset_ids: Sequence[str]) -> str:
    return sha256_hex(canonical_json(list(asset_ids)))


def price_vector_digest(asset_ids: Sequence[str], prices: np.ndarray) -> str:
    values = np.asarray(prices, dtype="<f8")
    return sha256_hex(
        canonical_json(list(asset_ids)),
        np.asarray(values.shape, dtype="<i8").tobytes(),
        values.tobytes(order="C"),
    )


def _as_id_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, str):
        parsed = json.loads(value)
        if not isinstance(parsed, list):
            raise ValueError(f"Expected JSON list, received {type(parsed).__name__}")
        return [str(item) for item in parsed]
    raise TypeError(f"Cannot interpret {type(value).__name__} as an identifier list")


@dataclass(frozen=True)
class ConstraintSnapshot:
    asset_ids: tuple[str, ...]
    matrix: sparse.csr_matrix
    bounds: np.ndarray
    row_metadata: pd.DataFrame
    epoch: int
    constraint_digest: str
    asset_digest: str

    @property
    def assets(self) -> int:
        return len(self.asset_ids)

    @property
    def constraints(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def nonzeros(self) -> int:
        return int(self.matrix.nnz)

    @property
    def density(self) -> float:
        denominator = self.matrix.shape[0] * self.matrix.shape[1]
        return float(self.matrix.nnz / denominator) if denominator else 0.0


def _snapshot_digest(
    asset_ids: Sequence[str],
    matrix: sparse.csr_matrix,
    bounds: np.ndarray,
    row_metadata: pd.DataFrame,
) -> str:
    # Builders already return sorted CSR matrices. Avoid copying the complete
    # sparse structure on the common path because digest construction is part
    # of the measured cold control-plane path.
    if not matrix.has_sorted_indices:
        matrix = matrix.copy()
        matrix.sort_indices()
    metadata_records = row_metadata.fillna("").to_dict(orient="records")
    return sha256_hex(
        canonical_json(list(asset_ids)),
        np.asarray(matrix.shape, dtype="<i8").tobytes(),
        np.asarray(matrix.indptr, dtype="<i8").tobytes(),
        np.asarray(matrix.indices, dtype="<i8").tobytes(),
        np.asarray(matrix.data, dtype="<f8").tobytes(),
        np.asarray(bounds, dtype="<f8").tobytes(),
        canonical_json(metadata_records),
    )


def _freeze_constraint_storage(
    matrix: sparse.csr_matrix,
    bounds: np.ndarray,
) -> None:
    """Make the safety-critical arrays read-only after snapshot construction."""

    for array in (matrix.data, matrix.indices, matrix.indptr, bounds):
        array.setflags(write=False)


def _validate_snapshot_identity(snapshot: "ConstraintSnapshot") -> None:
    expected_asset_digest = asset_order_digest(snapshot.asset_ids)
    if snapshot.asset_digest != expected_asset_digest:
        raise ValueError("snapshot asset digest does not match its asset order")
    expected_constraint_digest = _snapshot_digest(
        snapshot.asset_ids,
        snapshot.matrix,
        snapshot.bounds,
        snapshot.row_metadata,
    )
    if snapshot.constraint_digest != expected_constraint_digest:
        raise ValueError("snapshot constraint digest does not match its contents")


def _numeric_snapshot_integrity_digest(snapshot: "ConstraintSnapshot") -> str:
    """Hash the safety-critical numerical state for the hot commit path.

    Full metadata-inclusive identity is validated when a snapshot enters the
    store. Row metadata is diagnostic and does not participate in ``A p <= b``;
    re-serializing it at every commit needlessly dominated release latency.
    """

    matrix = snapshot.matrix
    return sha256_hex(
        snapshot.asset_digest.encode("ascii"),
        np.asarray(matrix.shape, dtype="<i8").tobytes(),
        np.asarray(matrix.indptr, dtype="<i8").tobytes(),
        np.asarray(matrix.indices, dtype="<i8").tobytes(),
        np.asarray(matrix.data, dtype="<f8").tobytes(),
        np.asarray(snapshot.bounds, dtype="<f8").tobytes(),
    )


def build_snapshot(
    asset_ids: Sequence[str],
    relations: pd.DataFrame,
    *,
    epoch: int,
    history_caps: Mapping[str, float] | None = None,
    include_nonnegative: bool = True,
) -> ConstraintSnapshot:
    """Compile sound relation records into ``A p <= b``.

    Expected relation types:

    * ``decomposition``: ``p(target) <= sum(p(component))``
    * ``projection_derivation`` / ``filter_derivation``:
      ``p(derived) <= p(source)``

    Only rows explicitly marked ``sound=True`` are accepted when a ``sound``
    column is present.
    """

    if isinstance(epoch, (bool, np.bool_)) or int(epoch) != epoch or int(epoch) < 0:
        raise ValueError("epoch must be a nonnegative integer")
    ordered_ids = tuple(str(item) for item in asset_ids)
    if not ordered_ids or any(not item for item in ordered_ids):
        raise ValueError("asset_ids must contain nonempty identifiers")
    if len(set(ordered_ids)) != len(ordered_ids):
        raise ValueError("asset_ids must be unique")
    index = {asset_id: position for position, asset_id in enumerate(ordered_ids)}

    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    bounds: list[float] = []
    metadata: list[dict[str, Any]] = []

    def add_row(
        coefficients: Mapping[str, float],
        bound: float,
        row_meta: Mapping[str, Any],
    ) -> None:
        if not coefficients:
            raise ValueError("constraint row must contain at least one coefficient")
        if not math.isfinite(float(bound)):
            raise ValueError("constraint bound must be finite")
        if any(not math.isfinite(float(value)) for value in coefficients.values()):
            raise ValueError("constraint coefficients must be finite")
        missing = [asset_id for asset_id in coefficients if asset_id not in index]
        if missing:
            raise KeyError(f"Constraint references unknown assets: {missing[:3]}")
        row = len(bounds)
        for asset_id, coefficient in coefficients.items():
            if coefficient == 0:
                continue
            row_indices.append(row)
            column_indices.append(index[asset_id])
            values.append(float(coefficient))
        bounds.append(float(bound))
        metadata.append(dict(row_meta))

    seen_relation_ids: set[str] = set()
    for record in relations.to_dict(orient="records"):
        if "sound" in record and not _strict_sound_flag(record["sound"]):
            continue
        relation_type = str(record["relation_type"])
        relation_id = str(record.get("relation_id", f"relation_{len(metadata)}"))
        if not relation_id:
            raise ValueError("relation_id must be nonempty")
        if relation_id in seen_relation_ids:
            raise ValueError(f"duplicate relation_id: {relation_id}")
        seen_relation_ids.add(relation_id)
        if relation_type == "decomposition":
            target = str(record["target_id"])
            components = _as_id_list(record["component_ids"])
            if not target:
                raise ValueError(f"{relation_id} has no target")
            if not components:
                raise ValueError(f"{relation_id} has no components")
            if any(not component for component in components):
                raise ValueError(f"{relation_id} has an empty component identifier")
            if len(set(components)) != len(components):
                raise ValueError(f"{relation_id} has duplicate components")
            if target in components:
                raise ValueError(f"{relation_id} contains its target as a component")
            coefficients: dict[str, float] = {target: 1.0}
            for component in components:
                coefficients[component] = coefficients.get(component, 0.0) - 1.0
            add_row(
                coefficients,
                0.0,
                {
                    "row_id": relation_id,
                    "constraint_type": relation_type,
                    "target_id": target,
                    "source_id": "",
                    "component_ids": json.dumps(components),
                },
            )
        elif relation_type in {"projection_derivation", "filter_derivation"}:
            derived = str(record["derived_id"])
            source = str(record["source_id"])
            if not derived or not source:
                raise ValueError(f"{relation_id} has an empty derived/source identifier")
            if derived == source:
                raise ValueError(f"{relation_id} derives an asset from itself")
            add_row(
                {derived: 1.0, source: -1.0},
                0.0,
                {
                    "row_id": relation_id,
                    "constraint_type": relation_type,
                    "target_id": derived,
                    "source_id": source,
                    "component_ids": "[]",
                },
            )
        else:
            raise ValueError(f"Unsupported relation_type: {relation_type}")

    if history_caps:
        for asset_id in sorted(history_caps):
            if asset_id not in index:
                raise KeyError(f"History cap references unknown asset: {asset_id}")
            cap = float(history_caps[asset_id])
            if not math.isfinite(cap) or cap < 0.0:
                raise ValueError("history caps must be finite and nonnegative")
            add_row(
                {asset_id: 1.0},
                cap,
                {
                    "row_id": f"history::{asset_id}",
                    "constraint_type": "buyer_history",
                    "target_id": asset_id,
                    "source_id": "",
                    "component_ids": "[]",
                },
            )

    if include_nonnegative:
        for asset_id in ordered_ids:
            add_row(
                {asset_id: -1.0},
                0.0,
                {
                    "row_id": f"nonnegative::{asset_id}",
                    "constraint_type": "nonnegative",
                    "target_id": asset_id,
                    "source_id": "",
                    "component_ids": "[]",
                },
            )

    matrix = sparse.coo_matrix(
        (values, (row_indices, column_indices)),
        shape=(len(bounds), len(ordered_ids)),
        dtype=np.float64,
    ).tocsr()
    matrix.sort_indices()
    if matrix.shape[0] and (np.diff(matrix.indptr) == 0).any():
        raise ValueError("every constraint row must contain a nonzero coefficient")
    bound_array = np.asarray(bounds, dtype=np.float64)
    row_metadata = pd.DataFrame(metadata).copy(deep=True)
    digest = _snapshot_digest(ordered_ids, matrix, bound_array, row_metadata)
    _freeze_constraint_storage(matrix, bound_array)
    return ConstraintSnapshot(
        asset_ids=ordered_ids,
        matrix=matrix,
        bounds=bound_array,
        row_metadata=row_metadata,
        epoch=int(epoch),
        constraint_digest=digest,
        asset_digest=asset_order_digest(ordered_ids),
    )


def extend_snapshot(
    snapshot: ConstraintSnapshot,
    new_relations: pd.DataFrame,
    *,
    new_epoch: int,
) -> ConstraintSnapshot:
    """Append relation rows without rebuilding unchanged rows.

    This models an incremental registry update. Asset additions or removals require
    a full rebuild because they change the vector order and matrix width.
    """

    if new_epoch <= snapshot.epoch:
        raise ValueError("new_epoch must be greater than the current epoch")
    additions = build_snapshot(
        snapshot.asset_ids,
        new_relations,
        epoch=new_epoch,
        include_nonnegative=False,
    )
    matrix = sparse.vstack([snapshot.matrix, additions.matrix], format="csr")
    bounds = np.concatenate([snapshot.bounds, additions.bounds])
    row_metadata = pd.concat(
        [snapshot.row_metadata, additions.row_metadata],
        ignore_index=True,
    )
    digest = _snapshot_digest(snapshot.asset_ids, matrix, bounds, row_metadata)
    _freeze_constraint_storage(matrix, bounds)
    return ConstraintSnapshot(
        asset_ids=snapshot.asset_ids,
        matrix=matrix,
        bounds=bounds,
        row_metadata=row_metadata,
        epoch=int(new_epoch),
        constraint_digest=digest,
        asset_digest=snapshot.asset_digest,
    )


def snapshot_without_rows(
    snapshot: ConstraintSnapshot,
    row_indices: Iterable[int],
    *,
    new_epoch: int | None = None,
) -> ConstraintSnapshot:
    """Return an immutable snapshot with explicitly selected rows omitted.

    This helper exists for the coverage-boundary experiment.  It records a new
    constraint digest while preserving the asset order, allowing the experiment
    to distinguish a vector that passes an incomplete snapshot from one that
    passes the complete supported-fragment snapshot.
    """

    omitted = {int(row) for row in row_indices}
    if any(row < 0 or row >= snapshot.constraints for row in omitted):
        raise IndexError("row_indices contains an out-of-range constraint row")
    keep = np.asarray(
        [row not in omitted for row in range(snapshot.constraints)],
        dtype=bool,
    )
    matrix = snapshot.matrix[keep].tocsr()
    matrix.sort_indices()
    bounds = snapshot.bounds[keep].copy()
    row_metadata = snapshot.row_metadata.loc[keep].reset_index(drop=True)
    digest = _snapshot_digest(
        snapshot.asset_ids,
        matrix,
        bounds,
        row_metadata,
    )
    _freeze_constraint_storage(matrix, bounds)
    return ConstraintSnapshot(
        asset_ids=snapshot.asset_ids,
        matrix=matrix,
        bounds=bounds,
        row_metadata=row_metadata,
        epoch=snapshot.epoch if new_epoch is None else int(new_epoch),
        constraint_digest=digest,
        asset_digest=snapshot.asset_digest,
    )


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    reason: str
    violations: int
    max_residual: float
    constraint_digest: str
    epoch: int


def independent_verify(
    snapshot: ConstraintSnapshot,
    prices: np.ndarray,
    *,
    ordered_asset_ids: Sequence[str] | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> VerificationResult:
    """Verify with a row-wise implementation separate from the repair kernel."""

    tolerance = _validated_tolerance(tolerance)
    vector = np.asarray(prices, dtype=np.float64)
    if vector.ndim != 1 or vector.shape[0] != snapshot.assets:
        return VerificationResult(
            False,
            "shape_mismatch",
            snapshot.constraints,
            math.inf,
            snapshot.constraint_digest,
            snapshot.epoch,
        )
    if ordered_asset_ids is not None:
        if tuple(str(item) for item in ordered_asset_ids) != snapshot.asset_ids:
            return VerificationResult(
                False,
                "asset_order_mismatch",
                snapshot.constraints,
                math.inf,
                snapshot.constraint_digest,
                snapshot.epoch,
            )
    if not np.isfinite(vector).all():
        return VerificationResult(
            False,
            "nonfinite_price",
            snapshot.constraints,
            math.inf,
            snapshot.constraint_digest,
            snapshot.epoch,
        )

    matrix = snapshot.matrix
    if matrix.shape[0] == 0:
        residuals = np.empty(0, dtype=np.float64)
    else:
        row_counts = np.diff(matrix.indptr)
        if (row_counts <= 0).any():
            raise ValueError("snapshot contains an empty constraint row")
        # Independent from the repair kernel's SciPy sparse-matrix multiply:
        # compute coefficient products and reduce each CSR segment in NumPy.
        products = matrix.data * vector[matrix.indices]
        row_sums = np.add.reduceat(products, matrix.indptr[:-1])
        residuals = row_sums - snapshot.bounds
    violations = int(np.count_nonzero(residuals > tolerance))
    maximum = float(residuals.max()) if residuals.size else 0.0
    return VerificationResult(
        passed=violations == 0,
        reason="verified" if violations == 0 else "constraint_violation",
        violations=violations,
        max_residual=maximum,
        constraint_digest=snapshot.constraint_digest,
        epoch=snapshot.epoch,
    )


@dataclass
class RepairResult:
    method: str
    prices: np.ndarray
    verification: VerificationResult
    rejected: bool
    fallback_used: bool
    solver: str
    sweeps: int
    projected_rows: int
    runtime_ms: float
    fallback_ms: float
    l1_adjustment: float
    error: str = ""


def _cyclic_projection(
    snapshot: ConstraintSnapshot,
    candidate: np.ndarray,
    *,
    active_only: bool,
    max_sweeps: int,
    active_row_budget: int,
    step: float,
    tolerance: float,
) -> tuple[np.ndarray, int, int]:
    prices = np.asarray(candidate, dtype=np.float64).copy()
    if not np.isfinite(prices).all():
        raise ValueError("candidate contains NaN or infinity")
    matrix = snapshot.matrix
    projected_rows = 0
    completed_sweeps = 0

    for sweep in range(max_sweeps):
        residuals = np.asarray(matrix @ prices - snapshot.bounds).ravel()
        violated = np.flatnonzero(residuals > tolerance)
        if violated.size == 0:
            break
        if active_only:
            order = np.argsort(-residuals[violated], kind="stable")
            rows = violated[order[:active_row_budget]]
        else:
            rows = np.arange(matrix.shape[0], dtype=np.int64)

        changed = 0
        for row in rows:
            start, end = matrix.indptr[row], matrix.indptr[row + 1]
            columns = matrix.indices[start:end]
            coefficients = matrix.data[start:end]
            residual = float(np.dot(coefficients, prices[columns]) - snapshot.bounds[row])
            if residual <= tolerance:
                continue
            squared_norm = float(np.dot(coefficients, coefficients))
            if squared_norm <= 0:
                continue
            prices[columns] -= step * residual * coefficients / squared_norm
            changed += 1
        completed_sweeps = sweep + 1
        projected_rows += changed
        if changed == 0:
            break
    return prices, completed_sweeps, projected_rows


def _osqp_projection(
    snapshot: ConstraintSnapshot,
    candidate: np.ndarray,
    *,
    tolerance: float,
    max_iter: int,
) -> tuple[np.ndarray, str, int]:
    vector = np.asarray(candidate, dtype=np.float64)
    if not np.isfinite(vector).all():
        raise ValueError("candidate contains NaN or infinity")

    if osqp is not None:
        identity = sparse.eye(snapshot.assets, format="csc", dtype=np.float64)
        constraints = snapshot.matrix.tocsc()
        solver = osqp.OSQP()
        solver.setup(
            P=identity,
            q=-vector,
            A=constraints,
            l=np.full(snapshot.constraints, -np.inf),
            u=snapshot.bounds,
            verbose=False,
            polishing=True,
            # A relative stopping term is inappropriate for the verifier:
            # controlled price units can differ by orders of magnitude while
            # release is governed by one absolute residual tolerance.
            eps_abs=max(min(1e-9, tolerance / 100), 1e-12),
            eps_rel=0.0,
            scaled_termination=True,
            max_iter=int(max_iter),
            adaptive_rho=True,
        )
        result = solver.solve()
        status = str(result.info.status).lower()
        if "solved" not in status or result.x is None:
            raise RuntimeError(f"OSQP did not solve the projection: {result.info.status}")
        return (
            np.asarray(result.x, dtype=np.float64),
            "osqp",
            int(result.info.iter),
        )

    # This fallback is for local validation only. Full Colab runs install OSQP.
    linear_constraint = LinearConstraint(
        snapshot.matrix,
        lb=np.full(snapshot.constraints, -np.inf),
        ub=snapshot.bounds,
    )
    result = minimize(
        fun=lambda x: 0.5 * float(np.square(x - vector).sum()),
        x0=vector,
        jac=lambda x: x - vector,
        constraints=[linear_constraint],
        method="SLSQP",
        options={"ftol": min(1e-10, tolerance / 10), "maxiter": max_iter},
    )
    if not result.success:
        raise RuntimeError(f"SciPy fallback did not solve the projection: {result.message}")
    return np.asarray(result.x, dtype=np.float64), "scipy_slsqp", int(result.nit)


def run_enforcement_method(
    snapshot: ConstraintSnapshot,
    candidate: np.ndarray,
    *,
    method: str,
    tolerance: float = DEFAULT_TOLERANCE,
    max_sweeps: int = 80,
    active_row_budget: int = 12_000,
    step: float = 1.0,
    allow_fallback: bool = True,
    osqp_max_iter: int = 20_000,
    force_fallback_failure: bool = False,
) -> RepairResult:
    """Run one enforcement method against one immutable snapshot."""

    tolerance = _validated_tolerance(tolerance)
    original = np.asarray(candidate, dtype=np.float64)
    start = time.perf_counter_ns()
    fallback_ms = 0.0
    fallback_used = False
    solver = method
    sweeps = 0
    projected_rows = 0
    error = ""

    try:
        if method == "reject_only":
            prices = original.copy()
        elif method == "arbiter_active":
            prices, sweeps, projected_rows = _cyclic_projection(
                snapshot,
                original,
                active_only=True,
                max_sweeps=max_sweeps,
                active_row_budget=active_row_budget,
                step=step,
                tolerance=tolerance,
            )
            solver = "active_cyclic"
        elif method == "full_cyclic":
            prices, sweeps, projected_rows = _cyclic_projection(
                snapshot,
                original,
                active_only=False,
                max_sweeps=max_sweeps,
                active_row_budget=snapshot.constraints,
                step=step,
                tolerance=tolerance,
            )
            solver = "full_cyclic"
        elif method == "osqp_qp":
            prices, solver, sweeps = _osqp_projection(
                snapshot,
                original,
                tolerance=tolerance,
                max_iter=osqp_max_iter,
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        verification = independent_verify(snapshot, prices, tolerance=tolerance)
        if (
            method == "arbiter_active"
            and not verification.passed
            and allow_fallback
        ):
            fallback_used = True
            fallback_start = time.perf_counter_ns()
            if force_fallback_failure:
                raise RuntimeError("injected fallback failure")
            prices, fallback_solver, fallback_iterations = _osqp_projection(
                snapshot,
                original,
                tolerance=tolerance,
                max_iter=osqp_max_iter,
            )
            fallback_ms = (time.perf_counter_ns() - fallback_start) / 1e6
            solver = f"active_cyclic+{fallback_solver}"
            sweeps += fallback_iterations
            verification = independent_verify(snapshot, prices, tolerance=tolerance)
    except Exception as exc:
        prices = original.copy()
        error = f"{type(exc).__name__}: {exc}"
        verification = VerificationResult(
            False,
            "solver_failure",
            snapshot.constraints,
            math.inf,
            snapshot.constraint_digest,
            snapshot.epoch,
        )

    runtime_ms = (time.perf_counter_ns() - start) / 1e6
    rejected = not verification.passed
    return RepairResult(
        method=method,
        prices=prices,
        verification=verification,
        rejected=rejected,
        fallback_used=fallback_used,
        solver=solver,
        sweeps=int(sweeps),
        projected_rows=int(projected_rows),
        runtime_ms=float(runtime_ms),
        fallback_ms=float(fallback_ms),
        l1_adjustment=float(np.abs(prices - original).sum()),
        error=error,
    )


def exact_box_row_attacks(
    snapshot: ConstraintSnapshot,
    feasible_center: np.ndarray,
    *,
    budget_fraction: float,
    limit: int,
    tolerance: float = DEFAULT_TOLERANCE,
    include_types: Iterable[str] = (
        "decomposition",
        "projection_derivation",
        "filter_derivation",
    ),
) -> list[dict[str, Any]]:
    """Construct exact per-row attacks under coordinate-wise price bounds.

    For each selected constraint row, the linear residual is maximized exactly
    over ``center_i +/- budget_fraction * max(|center_i|, median(|center|))``.
    This certifies the strongest attack for that row and box, but it is not a
    certificate for maximizing the sum of violations over all rows.
    """

    tolerance = _validated_tolerance(tolerance)
    budget_fraction = float(budget_fraction)
    if not math.isfinite(budget_fraction) or budget_fraction <= 0.0:
        raise ValueError("budget_fraction must be finite and strictly positive")
    if isinstance(limit, (bool, np.bool_)) or int(limit) != limit or int(limit) < 1:
        raise ValueError("limit must be a positive integer")
    center = np.asarray(feasible_center, dtype=np.float64)
    base_check = independent_verify(snapshot, center, tolerance=tolerance)
    if not base_check.passed:
        raise ValueError("exact_box_row_attacks requires a verified center")
    scale = max(float(np.median(np.abs(center))), tolerance)
    widths = budget_fraction * np.maximum(np.abs(center), scale)
    lower = np.maximum(0.0, center - widths)
    upper = center + widths
    allowed = set(include_types)

    ranked_rows: list[dict[str, Any]] = []
    for row, metadata in snapshot.row_metadata.iterrows():
        if metadata["constraint_type"] not in allowed:
            continue
        start, end = snapshot.matrix.indptr[row], snapshot.matrix.indptr[row + 1]
        columns = snapshot.matrix.indices[start:end]
        coefficients = snapshot.matrix.data[start:end]
        extremal_values = np.where(
            coefficients >= 0, upper[columns], lower[columns]
        )
        row_residual = float(
            np.dot(coefficients, extremal_values) - snapshot.bounds[row]
        )
        ranked_rows.append(
            {
                "row_index": int(row),
                "row_id": str(metadata["row_id"]),
                "constraint_type": str(metadata["constraint_type"]),
                "budget_fraction": float(budget_fraction),
                "certified_target_residual": row_residual,
            }
        )
    ranked_rows.sort(
        key=lambda record: (
            record["certified_target_residual"],
            record["row_id"],
        ),
        reverse=True,
    )

    attacks: list[dict[str, Any]] = []
    # Rows whose exact box optimum does not cross the verifier tolerance are
    # not attacks and must not inflate campaign sample sizes.
    attackable_rows = [
        record
        for record in ranked_rows
        if record["certified_target_residual"] > tolerance
    ]
    for record in attackable_rows[:limit]:
        row = int(record["row_index"])
        start, end = snapshot.matrix.indptr[row], snapshot.matrix.indptr[row + 1]
        columns = snapshot.matrix.indices[start:end]
        coefficients = snapshot.matrix.data[start:end]
        candidate = center.copy()
        candidate[columns] = np.where(
            coefficients >= 0, upper[columns], lower[columns]
        )
        full_residuals = np.asarray(snapshot.matrix @ candidate - snapshot.bounds)
        record = dict(record)
        record.update(
            {
                "violations_before": int((full_residuals > tolerance).sum()),
                "max_residual_before": float(
                    full_residuals.max(initial=-math.inf)
                ),
                "candidate": candidate,
            }
        )
        attacks.append(record)
    return attacks


@dataclass(frozen=True)
class ReleaseToken:
    token_id: str
    epoch: int
    constraint_digest: str
    asset_digest: str
    price_digest: str
    nonce: str
    verifier_tolerance: float
    mac: str

    def signed_fields(self) -> dict[str, Any]:
        return {
            "token_id": self.token_id,
            "epoch": self.epoch,
            "constraint_digest": self.constraint_digest,
            "asset_digest": self.asset_digest,
            "price_digest": self.price_digest,
            "nonce": self.nonce,
            "verifier_tolerance": self.verifier_tolerance,
        }


@dataclass
class AnswerBuffer:
    payload: Any
    released: bool = False
    retained_reason: str = ""

    def release(self) -> Any:
        self.released = True
        self.retained_reason = ""
        return self.payload

    def retain(self, reason: str) -> None:
        self.released = False
        self.retained_reason = reason


@dataclass(frozen=True)
class CommitResult:
    released: bool
    reason: str
    epoch: int
    token_id: str


class SnapshotStore:
    """In-process prototype of immutable snapshots and atomic compare-and-commit."""

    def __init__(
        self,
        initial_snapshot: ConstraintSnapshot,
        secret: bytes | None = None,
        *,
        verifier_tolerance: float = DEFAULT_TOLERANCE,
    ):
        _validate_snapshot_identity(initial_snapshot)
        self._snapshot = initial_snapshot
        self._numeric_integrity_digest = _numeric_snapshot_integrity_digest(
            initial_snapshot
        )
        self._secret = secret if secret is not None else secrets.token_bytes(32)
        if not isinstance(self._secret, bytes) or len(self._secret) < 16:
            raise ValueError("secret must contain at least 16 bytes")
        self._verifier_tolerance = _validated_tolerance(verifier_tolerance)
        self._lock = threading.RLock()
        self._committed_tokens: set[str] = set()

    def read(self) -> ConstraintSnapshot:
        with self._lock:
            return self._snapshot

    def advance(self, snapshot: ConstraintSnapshot) -> None:
        with self._lock:
            if snapshot.epoch <= self._snapshot.epoch:
                raise ValueError("new snapshot epoch must increase")
            _validate_snapshot_identity(snapshot)
            self._snapshot = snapshot
            self._numeric_integrity_digest = _numeric_snapshot_integrity_digest(
                snapshot
            )

    def restart_with_new_secret(self) -> None:
        with self._lock:
            self._secret = secrets.token_bytes(32)
            self._committed_tokens.clear()

    def _token_mac(self, signed_fields: Mapping[str, Any]) -> str:
        return hmac.new(
            self._secret,
            canonical_json(dict(signed_fields)),
            hashlib.sha256,
        ).hexdigest()

    def issue_token(
        self,
        snapshot: ConstraintSnapshot,
        prices: np.ndarray,
        *,
        tolerance: float = DEFAULT_TOLERANCE,
    ) -> ReleaseToken:
        tolerance = _validated_tolerance(tolerance)
        if tolerance != self._verifier_tolerance:
            raise ValueError("requested tolerance does not match the store policy")
        verification = independent_verify(snapshot, prices, tolerance=tolerance)
        if not verification.passed:
            raise ValueError("cannot issue a release token for an unverified vector")
        fields = {
            "token_id": secrets.token_hex(16),
            "epoch": snapshot.epoch,
            "constraint_digest": snapshot.constraint_digest,
            "asset_digest": snapshot.asset_digest,
            "price_digest": price_vector_digest(snapshot.asset_ids, prices),
            "nonce": secrets.token_hex(16),
            "verifier_tolerance": float(tolerance),
        }
        return ReleaseToken(mac=self._token_mac(fields), **fields)

    def compare_and_commit(
        self,
        token: ReleaseToken,
        prices: np.ndarray,
        buffer: AnswerBuffer,
    ) -> CommitResult:
        """Atomically compare current state, consume the token, and release."""

        with self._lock:
            current = self._snapshot
            if (
                _numeric_snapshot_integrity_digest(current)
                != self._numeric_integrity_digest
            ):
                buffer.retain("snapshot_integrity_failure")
                return CommitResult(
                    False,
                    "snapshot_integrity_failure",
                    current.epoch,
                    token.token_id,
                )
            if token.token_id in self._committed_tokens:
                buffer.retain("token_replay")
                return CommitResult(False, "token_replay", current.epoch, token.token_id)
            if token.epoch != current.epoch:
                buffer.retain("stale_epoch")
                return CommitResult(False, "stale_epoch", current.epoch, token.token_id)
            if token.constraint_digest != current.constraint_digest:
                buffer.retain("constraint_digest_mismatch")
                return CommitResult(
                    False,
                    "constraint_digest_mismatch",
                    current.epoch,
                    token.token_id,
                )
            if token.asset_digest != current.asset_digest:
                buffer.retain("asset_digest_mismatch")
                return CommitResult(
                    False,
                    "asset_digest_mismatch",
                    current.epoch,
                    token.token_id,
                )
            try:
                token_tolerance = _validated_tolerance(token.verifier_tolerance)
                expected_mac = self._token_mac(token.signed_fields())
            except Exception:
                buffer.retain("invalid_token_fields")
                return CommitResult(
                    False,
                    "invalid_token_fields",
                    current.epoch,
                    token.token_id,
                )
            if token_tolerance != self._verifier_tolerance:
                buffer.retain("tolerance_policy_mismatch")
                return CommitResult(
                    False,
                    "tolerance_policy_mismatch",
                    current.epoch,
                    token.token_id,
                )
            if not hmac.compare_digest(token.mac, expected_mac):
                buffer.retain("token_mac_mismatch")
                return CommitResult(
                    False,
                    "token_mac_mismatch",
                    current.epoch,
                    token.token_id,
                )
            actual_price_digest = price_vector_digest(current.asset_ids, prices)
            if token.price_digest != actual_price_digest:
                buffer.retain("price_digest_mismatch")
                return CommitResult(
                    False,
                    "price_digest_mismatch",
                    current.epoch,
                    token.token_id,
                )
            verification = independent_verify(
                current,
                prices,
                ordered_asset_ids=current.asset_ids,
                tolerance=token_tolerance,
            )
            if not verification.passed:
                buffer.retain("commit_verification_failed")
                return CommitResult(
                    False,
                    "commit_verification_failed",
                    current.epoch,
                    token.token_id,
                )
            self._committed_tokens.add(token.token_id)
            buffer.release()
            return CommitResult(True, "released", current.epoch, token.token_id)


def replace_token(token: ReleaseToken, **changes: Any) -> ReleaseToken:
    """Fault-injection helper. Any changed signed field invalidates the MAC."""

    return replace(token, **changes)


def execute_buffer_verify_commit(
    *,
    executor: Callable[[str], Any],
    sql: str,
    store: SnapshotStore,
    proposed_prices: np.ndarray,
    method: str = "arbiter_active",
    tolerance: float = DEFAULT_TOLERANCE,
    before_commit: Callable[[], None] | None = None,
    **repair_options: Any,
) -> tuple[AnswerBuffer, RepairResult, ReleaseToken | None, CommitResult]:
    """Execute a query, hold its answer, then release only after atomic commit."""

    answer = executor(sql)
    buffer = AnswerBuffer(answer)
    snapshot = store.read()
    repair = run_enforcement_method(
        snapshot,
        proposed_prices,
        method=method,
        tolerance=tolerance,
        **repair_options,
    )
    if repair.rejected:
        buffer.retain(repair.verification.reason)
        return (
            buffer,
            repair,
            None,
            CommitResult(False, repair.verification.reason, snapshot.epoch, ""),
        )
    token = store.issue_token(snapshot, repair.prices, tolerance=tolerance)
    if before_commit is not None:
        before_commit()
    commit = store.compare_and_commit(token, repair.prices, buffer)
    return buffer, repair, token, commit


def median_and_quantiles(
    values: Sequence[float],
    quantiles: Sequence[float] = (0.5, 0.95, 0.99),
) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {f"p{int(q * 100):02d}": math.nan for q in quantiles}
    return {
        f"p{int(q * 100):02d}": float(np.quantile(array, q, method="linear"))
        for q in quantiles
    }


def environment_manifest(extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    import importlib.metadata
    import platform
    import sys

    packages: dict[str, str] = {}
    for package in [
        "duckdb",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "osqp",
        "sqlglot",
        "pyarrow",
        "matplotlib",
    ]:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "not-installed"
    result: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": __import__("os").cpu_count(),
        "packages": packages,
    }
    if extra:
        result["config"] = dict(extra)
    return result


def checksum_manifest(directory: str) -> pd.DataFrame:
    from pathlib import Path

    root = Path(directory)
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name in {"checksums.sha256", "checksums.csv"}:
            continue
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return pd.DataFrame(rows)
