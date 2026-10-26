"""Targeted experiments for the ARBITER request-binding closure campaign."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import secrets
import sys
import tempfile
import threading
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

import arbiter_core as core
from arbiter_experiments import relation_family_attack_center
from arbiter_request_release import (
    BoundAnswerBuffer,
    DurableReleaseLedger,
    RequestBoundSnapshotStore,
    RequestContext,
    exact_sql_digest,
    next_epoch_snapshot,
    replace_bound_token,
)


AUDITED_SUITE_VERSION = "2026-08-01-audited-v4"
CLOSURE_SUITE_VERSION = "2026-08-01-request-bound-closure-v1.1"
AUDITED_CODE_HASHES = {
    "arbiter_core.py": "b6797381dd883eac5d3cef7f1f86eaa2ad66bf73c498fdcfb7493f5113175ffa",
    "arbiter_data.py": "fb4d85f6a36be12bf1dc3ac09e80383f4d7e52e83a923bae1840a87443452c86",
    "arbiter_experiments.py": "f5919e83b4c2658950a611e94f3bc8de62000a4ab7bd1cb24b90f64e5a0b590e",
    "arbiter_workload.py": "cd3a78b28c573698e77fe02bc301273bbc959d770ee6daadae52ace68151ef5a",
}
RELATION_FAMILIES = (
    "decomposition",
    "projection_derivation",
    "filter_derivation",
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


def verify_audited_inputs(
    run_dir: Path,
    code_dir: Path,
    *,
    active_run_dir: Path | None = None,
) -> pd.DataFrame:
    """Fail closed on stale, partial, modified, or non-full audited inputs.

    ``run_dir`` is the immutable evidence-package root created by Notebook 04.
    When ``active_run_dir`` is supplied, every packaged stage file is also
    checked against the live 00--04 run.  This prevents a stale nested package
    from being accepted merely because it is internally self-consistent.
    """

    run_dir = run_dir.resolve()
    code_dir = code_dir.resolve()
    config_path = run_dir / "run_config.json"
    checksums_path = run_dir / "checksums.csv"
    quality_path = run_dir / "04_evidence" / "all_stage_quality_checks.csv"
    for required in [config_path, checksums_path, quality_path]:
        if not required.is_file():
            raise FileNotFoundError(f"required audited input is missing: {required}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("run_mode") != "full":
        raise RuntimeError("closure evidence requires a completed full run")
    if config.get("suite_version") != AUDITED_SUITE_VERSION:
        raise RuntimeError("closure input is not the audited-v4 suite")

    quality = pd.read_csv(quality_path)
    passed = quality["passed"].astype(str).str.lower().map({"true": True, "false": False})
    if passed.isna().any() or not passed.all() or len(quality) != 52:
        raise RuntimeError("the prerequisite 52 audited-v4 stage checks do not all pass")

    rows: list[dict[str, Any]] = []
    manifest = pd.read_csv(checksums_path)
    required_columns = {"path", "bytes", "sha256"}
    if not required_columns.issubset(manifest.columns):
        raise RuntimeError("audited checksum manifest has an unexpected schema")
    if manifest.empty:
        raise RuntimeError("audited checksum manifest is empty")
    if manifest["path"].isna().any() or manifest["path"].astype(str).duplicated().any():
        raise RuntimeError("audited checksum manifest has missing or duplicate paths")
    if manifest["bytes"].isna().any():
        raise RuntimeError("audited checksum manifest has missing byte counts")
    byte_counts = pd.to_numeric(manifest["bytes"], errors="coerce")
    if byte_counts.isna().any() or (byte_counts < 0).any() or not np.all(
        byte_counts.to_numpy(float) == np.floor(byte_counts.to_numpy(float))
    ):
        raise RuntimeError("audited checksum manifest has invalid byte counts")
    digests = manifest["sha256"].astype(str)
    if not digests.str.fullmatch(r"[0-9a-f]{64}").all():
        raise RuntimeError("audited checksum manifest has invalid SHA-256 values")

    sha_text_path = run_dir / "checksums.sha256"
    if not sha_text_path.is_file():
        raise FileNotFoundError(
            f"required audited input is missing: {sha_text_path}"
        )
    expected_sha_text = "".join(
        f"{row.sha256}  {row.path}\n" for row in manifest.itertuples(index=False)
    )
    if sha_text_path.read_text(encoding="utf-8") != expected_sha_text:
        raise RuntimeError("audited CSV and SHA-256 manifests disagree")

    for record in manifest.itertuples(index=False):
        relative = Path(str(record.path))
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"unsafe checksum path: {relative}")
        target = (run_dir / relative).resolve()
        if run_dir not in target.parents:
            raise RuntimeError(f"checksum path escaped run directory: {relative}")
        exists = target.is_file()
        observed_bytes = target.stat().st_size if exists else -1
        observed_hash = sha256_file(target) if exists else ""
        row_passed = bool(
            exists
            and observed_bytes == int(record.bytes)
            and observed_hash == str(record.sha256)
        )
        rows.append(
            {
                "scope": "run_evidence",
                "path": relative.as_posix(),
                "expected_bytes": int(record.bytes),
                "observed_bytes": int(observed_bytes),
                "expected_sha256": str(record.sha256),
                "observed_sha256": observed_hash,
                "passed": row_passed,
            }
        )

    if active_run_dir is not None:
        active_run_dir = active_run_dir.resolve()
        if not active_run_dir.is_dir():
            raise FileNotFoundError(
                f"active audited run directory is missing: {active_run_dir}"
            )
        if active_run_dir != run_dir:
            live_roots = {
                "00_workload",
                "01_proposals_attacks",
                "02_release_protocol",
                "03_scalability",
                "04_evidence",
            }
            linked_records = 0
            for record in manifest.itertuples(index=False):
                relative = Path(str(record.path))
                is_live_output = (
                    relative.as_posix() == "run_config.json"
                    or (relative.parts and relative.parts[0] in live_roots)
                )
                if not is_live_output:
                    continue
                linked_records += 1
                target = (active_run_dir / relative).resolve()
                if active_run_dir not in target.parents:
                    raise RuntimeError(
                        f"active-run checksum path escaped its directory: {relative}"
                    )
                exists = target.is_file()
                observed_bytes = target.stat().st_size if exists else -1
                observed_hash = sha256_file(target) if exists else ""
                row_passed = bool(
                    exists
                    and observed_bytes == int(record.bytes)
                    and observed_hash == str(record.sha256)
                )
                rows.append(
                    {
                        "scope": "active_run_package_link",
                        "path": relative.as_posix(),
                        "expected_bytes": int(record.bytes),
                        "observed_bytes": int(observed_bytes),
                        "expected_sha256": str(record.sha256),
                        "observed_sha256": observed_hash,
                        "passed": row_passed,
                    }
                )
            if linked_records == 0:
                raise RuntimeError(
                    "audited manifest contains no live 00-04 outputs to link"
                )

    for name, expected_hash in AUDITED_CODE_HASHES.items():
        target = code_dir / name
        observed_hash = sha256_file(target) if target.is_file() else ""
        rows.append(
            {
                "scope": "pinned_code",
                "path": name,
                "expected_bytes": None,
                "observed_bytes": target.stat().st_size if target.is_file() else -1,
                "expected_sha256": expected_hash,
                "observed_sha256": observed_hash,
                "passed": observed_hash == expected_hash,
            }
        )
    audit = pd.DataFrame(rows)
    if not audit["passed"].all():
        failed = audit.loc[~audit["passed"], "path"].tolist()
        raise RuntimeError(f"audited input integrity failed: {failed[:5]}")
    return audit


@dataclass
class DatasetBundle:
    dataset: str
    assets: pd.DataFrame
    relations: pd.DataFrame
    snapshot: core.ConstraintSnapshot
    prices: np.ndarray
    query_digests: dict[str, str]


@dataclass
class TieredRepairResult:
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


def run_tiered_fail_safe_repair(
    snapshot: core.ConstraintSnapshot,
    candidate: np.ndarray,
    *,
    tolerance: float,
    max_sweeps: int,
    active_row_budget: int,
    step: float,
    osqp_max_iter: int,
    full_fallback_sweeps: int = 20,
) -> TieredRepairResult:
    """Use the cheapest verified repair tier before invoking a QP solver.

    Dense cases showed that constructing and iterating OSQP can dominate a
    fallback even when two full cyclic sweeps suffice. Every tier is followed by
    the independent verifier; a failed final tier is rejected.
    """

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
        return TieredRepairResult(
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

    full = core.run_enforcement_method(
        snapshot,
        active.prices,
        method="full_cyclic",
        tolerance=tolerance,
        max_sweeps=full_fallback_sweeps,
        active_row_budget=snapshot.constraints,
        step=step,
        allow_fallback=False,
        osqp_max_iter=osqp_max_iter,
    )
    total_runtime = active.runtime_ms + full.runtime_ms
    total_sweeps = active.sweeps + full.sweeps
    total_projected = active.projected_rows + full.projected_rows
    if full.verification.passed:
        return TieredRepairResult(
            prices=full.prices,
            verification=full.verification,
            rejected=False,
            fallback_used=True,
            fallback_stage="full_cyclic",
            solver="active_cyclic+full_cyclic",
            sweeps=total_sweeps,
            projected_rows=total_projected,
            runtime_ms=total_runtime,
            fallback_ms=full.runtime_ms,
            l1_adjustment=float(np.abs(full.prices - original).sum()),
            error="; ".join(part for part in [active.error, full.error] if part),
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
    total_runtime += qp.runtime_ms
    final_verification = core.independent_verify(
        snapshot, qp.prices, tolerance=tolerance
    )
    rejected = not final_verification.passed
    return TieredRepairResult(
        prices=qp.prices,
        verification=final_verification,
        rejected=rejected,
        fallback_used=True,
        fallback_stage="osqp" if not rejected else "rejected_after_osqp",
        solver="active_cyclic+full_cyclic+osqp",
        sweeps=total_sweeps + qp.sweeps,
        projected_rows=total_projected + qp.projected_rows,
        runtime_ms=total_runtime,
        fallback_ms=full.runtime_ms + qp.runtime_ms,
        l1_adjustment=float(np.abs(qp.prices - original).sum()),
        error="; ".join(
            part for part in [active.error, full.error, qp.error] if part
        ),
    )


def build_dataset_bundles(
    assets: pd.DataFrame,
    relations: pd.DataFrame,
    policy: pd.DataFrame,
    *,
    tolerance: float,
) -> dict[str, DatasetBundle]:
    supported = assets[
        assets["expected_supported"].astype(str).str.lower().eq("true")
    ].copy()
    if supported.duplicated(["dataset", "asset_id"]).any():
        raise ValueError("supported asset identifiers are not unique per dataset")

    bundles: dict[str, DatasetBundle] = {}
    for dataset, dataset_assets in supported.groupby("dataset", sort=False):
        dataset_assets = dataset_assets.reset_index(drop=True)
        asset_ids = dataset_assets["asset_id"].astype(str).tolist()
        dataset_relations = relations[relations["dataset"] == dataset].copy()
        snapshot = core.build_snapshot(asset_ids, dataset_relations, epoch=1)
        price_rows = policy[policy["dataset"] == dataset][
            ["asset_id", "target_price"]
        ]
        if price_rows["asset_id"].duplicated().any():
            raise ValueError(f"duplicate controlled prices for {dataset}")
        price_map = price_rows.set_index("asset_id")["target_price"].to_dict()
        if set(price_map) != set(asset_ids):
            raise ValueError(f"controlled price coverage mismatch for {dataset}")
        prices = np.asarray([price_map[asset_id] for asset_id in asset_ids], dtype=np.float64)
        verification = core.independent_verify(snapshot, prices, tolerance=tolerance)
        if not verification.passed:
            raise RuntimeError(f"controlled policy is infeasible for {dataset}")
        query_digests = {
            str(row.asset_id): exact_sql_digest(str(row.sql))
            for row in dataset_assets.itertuples(index=False)
        }
        bundles[str(dataset)] = DatasetBundle(
            dataset=str(dataset),
            assets=dataset_assets,
            relations=dataset_relations,
            snapshot=snapshot,
            prices=prices,
            query_digests=query_digests,
        )
    if len(bundles) != 3:
        raise RuntimeError(f"expected three audited datasets, found {len(bundles)}")
    return bundles


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:80] or "dataset"


def _protocol_assets(bundle: DatasetBundle) -> tuple[pd.Series, pd.Series]:
    scalar = bundle.assets[
        bundle.assets["family"].astype(str).str.contains("aggregate_partition", na=False)
    ].sort_values("asset_id")
    if len(scalar) < 2:
        raise RuntimeError(f"{bundle.dataset} lacks two scalar protocol assets")
    return scalar.iloc[0], scalar.iloc[1]


def _request_for_row(
    row: pd.Series,
    prices_by_asset: Mapping[str, float],
    *,
    request_id: str,
    buyer_id: str = "buyer-a",
    session_id: str = "session-a",
) -> RequestContext:
    asset_id = str(row["asset_id"])
    return RequestContext.from_values(
        request_id=request_id,
        buyer_id=buyer_id,
        session_id=session_id,
        asset_id=asset_id,
        sql=str(row["sql"]),
        quoted_price=float(prices_by_asset[asset_id]),
    )


class _MutableClock:
    def __init__(self, value: int) -> None:
        self.value = int(value)

    def __call__(self) -> int:
        return self.value


def run_protocol_fault_campaign(
    bundles: Mapping[str, DatasetBundle],
    *,
    answer_executor: Callable[[str, str], Any],
    work_dir: Path,
    tolerance: float,
    concurrency_trials: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Exercise field substitutions, replay, mutation, expiry, and races."""

    work_dir.mkdir(parents=True, exist_ok=True)
    fault_rows: list[dict[str, Any]] = []
    concurrency_rows: list[dict[str, Any]] = []
    delivery_rows: list[dict[str, Any]] = []
    durable_rows: list[dict[str, Any]] = []

    def record(
        *,
        dataset: str,
        case: str,
        expected_release: bool,
        result,
        buffer: BoundAnswerBuffer | None,
        ledger: DurableReleaseLedger,
        expected_reasons: Iterable[str] = (),
        phase: str = "commit",
    ) -> None:
        release_count = len(ledger.release_rows())
        observed_release = bool(result.released) if result is not None else False
        reason = str(result.reason) if result is not None else "issuance_rejected"
        reason_ok = not expected_reasons or reason in set(expected_reasons)
        passed = bool(
            observed_release == expected_release
            and reason_ok
            and release_count == (1 if expected_release else 0)
            and (
                buffer is None
                or bool(buffer.released) == expected_release
                or (case == "token_replay" and buffer.released)
            )
        )
        fault_rows.append(
            {
                "dataset": dataset,
                "case": case,
                "phase": phase,
                "expected_release": expected_release,
                "released": observed_release,
                "durable": bool(result.durable) if result is not None else False,
                "reason": reason,
                "buffer_released": bool(buffer.released) if buffer is not None else False,
                "durable_release_rows": release_count,
                "passed": passed,
            }
        )

    for dataset, bundle in bundles.items():
        dataset_dir = work_dir / _safe_slug(dataset)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        row_a, row_b = _protocol_assets(bundle)
        price_map = dict(zip(bundle.snapshot.asset_ids, bundle.prices, strict=True))
        answer_a = answer_executor(dataset, str(row_a["sql"]))
        answer_b = answer_executor(dataset, str(row_b["sql"]))
        if answer_a == answer_b:
            # Distinct query/asset bindings remain meaningful even if two scalar
            # aggregate values happen to coincide. The answer-swap case below
            # uses a typed wrapper so its digest is certainly distinct.
            answer_b = {"asset_id": str(row_b["asset_id"]), "rows": answer_b}

        case_counter = 0

        def prepared(
            label: str,
            *,
            clock: Callable[[], int] | None = None,
            ttl_ns: int = 10_000_000_000,
            secret: bytes | None = None,
        ):
            nonlocal case_counter
            case_counter += 1
            ledger_path = dataset_dir / f"{case_counter:03d}-{_safe_slug(label)}.sqlite"
            if ledger_path.exists():
                ledger_path.unlink()
            ledger = DurableReleaseLedger(ledger_path)
            service_secret = secret if secret is not None else secrets.token_bytes(32)
            store = RequestBoundSnapshotStore(
                bundle.snapshot,
                ledger=ledger,
                asset_query_digests=bundle.query_digests,
                secret=service_secret,
                verifier_tolerance=tolerance,
                token_ttl_ns=ttl_ns,
                clock_ns=clock if clock is not None else time.time_ns,
            )
            request = _request_for_row(
                row_a,
                price_map,
                request_id=f"{_safe_slug(dataset)}-{label}-{case_counter}",
            )
            buffer = BoundAnswerBuffer(request=request, payload=answer_a)
            token = store.issue_token(bundle.snapshot, bundle.prices, request, buffer)
            return ledger, store, request, buffer, token, service_secret

        # Positive control.
        ledger, store, request, buffer, token, _ = prepared("normal")
        result = store.compare_and_commit(token, bundle.prices, request, buffer)
        record(
            dataset=dataset,
            case="normal_request_bound_release",
            expected_release=True,
            result=result,
            buffer=buffer,
            ledger=ledger,
            expected_reasons={"released"},
        )
        recovered = ledger.released_payload(token_id=token.token_id, request=request)
        delivery_rows.append(
            {
                "dataset": dataset,
                "case": "authorized_delivery",
                "delivered": recovered == buffer.payload_bytes,
                "reason": "delivered" if recovered == buffer.payload_bytes else "payload_mismatch",
                "passed": recovered == buffer.payload_bytes,
            }
        )
        wrong_delivery_request = replace(request, buyer_id="buyer-substituted")
        try:
            ledger.released_payload(token_id=token.token_id, request=wrong_delivery_request)
            wrong_delivery_passed = False
            wrong_delivery_reason = "unexpected_delivery"
        except PermissionError as exc:
            wrong_delivery_passed = True
            wrong_delivery_reason = str(exc)
        delivery_rows.append(
            {
                "dataset": dataset,
                "case": "cross_buyer_delivery",
                "delivered": False,
                "reason": wrong_delivery_reason,
                "passed": wrong_delivery_passed,
            }
        )
        durable_rows.extend({"dataset": dataset, **row} for row in ledger.release_rows())

        substitution_builders = {
            "request_id_swap": lambda original: replace(original, request_id=original.request_id + "-swap"),
            "buyer_id_swap": lambda original: replace(original, buyer_id="buyer-b"),
            "session_id_swap": lambda original: replace(original, session_id="session-b"),
            "query_swap": lambda original: replace(
                original, query_digest=exact_sql_digest(str(row_b["sql"]))
            ),
            "quoted_price_swap": lambda original: replace(
                original,
                quoted_price_hex=float(np.nextafter(original.quoted_price, math.inf)).hex(),
            ),
        }
        for label, builder in substitution_builders.items():
            ledger, store, request, buffer, token, _ = prepared(label)
            substituted_request = builder(request)
            substituted_buffer = BoundAnswerBuffer(
                request=substituted_request,
                payload=answer_a,
            )
            result = store.compare_and_commit(
                token, bundle.prices, substituted_request, substituted_buffer
            )
            record(
                dataset=dataset,
                case=label,
                expected_release=False,
                result=result,
                buffer=substituted_buffer,
                ledger=ledger,
            )

        # Cross-asset request contains the registered SQL and quote for asset B.
        ledger, store, request, buffer, token, _ = prepared("asset-swap")
        request_b = _request_for_row(
            row_b,
            price_map,
            request_id=request.request_id,
            buyer_id=request.buyer_id,
            session_id=request.session_id,
        )
        buffer_b = BoundAnswerBuffer(request=request_b, payload=answer_b)
        result = store.compare_and_commit(token, bundle.prices, request_b, buffer_b)
        record(
            dataset=dataset,
            case="asset_swap",
            expected_release=False,
            result=result,
            buffer=buffer_b,
            ledger=ledger,
        )

        # Same request metadata but a different answer/buffer.
        ledger, store, request, buffer, token, _ = prepared("answer-swap")
        swapped_buffer = BoundAnswerBuffer(
            request=request,
            payload={"substituted": answer_b},
        )
        result = store.compare_and_commit(token, bundle.prices, request, swapped_buffer)
        record(
            dataset=dataset,
            case="answer_buffer_swap",
            expected_release=False,
            result=result,
            buffer=swapped_buffer,
            ledger=ledger,
            expected_reasons={"buffer_id_mismatch"},
        )

        # Valid token A used against complete request/buffer B.
        ledger, store, request, buffer, token, _ = prepared("cross-request-token")
        request_b = _request_for_row(
            row_b,
            price_map,
            request_id=request.request_id + "-b",
            buyer_id="buyer-b",
            session_id="session-b",
        )
        buffer_b = BoundAnswerBuffer(request=request_b, payload=answer_b)
        result = store.compare_and_commit(token, bundle.prices, request_b, buffer_b)
        record(
            dataset=dataset,
            case="cross_request_token_swap",
            expected_release=False,
            result=result,
            buffer=buffer_b,
            ledger=ledger,
        )

        ledger, store, request, buffer, token, _ = prepared("price-vector-swap")
        changed_prices = bundle.prices.copy()
        changed_prices[1] = np.nextafter(changed_prices[1], math.inf)
        result = store.compare_and_commit(token, changed_prices, request, buffer)
        record(
            dataset=dataset,
            case="price_vector_swap",
            expected_release=False,
            result=result,
            buffer=buffer,
            ledger=ledger,
            expected_reasons={"price_digest_mismatch"},
        )

        ledger, store, request, buffer, token, _ = prepared("signed-field-tamper")
        tampered = replace_bound_token(token, buyer_id="tampered-buyer")
        result = store.compare_and_commit(tampered, bundle.prices, request, buffer)
        record(
            dataset=dataset,
            case="signed_field_tamper",
            expected_release=False,
            result=result,
            buffer=buffer,
            ledger=ledger,
            expected_reasons={"token_mac_mismatch"},
        )

        ledger, store, request, buffer, token, _ = prepared("mac-tamper")
        tampered = replace_bound_token(token, mac="0" * 64)
        result = store.compare_and_commit(tampered, bundle.prices, request, buffer)
        record(
            dataset=dataset,
            case="mac_tamper",
            expected_release=False,
            result=result,
            buffer=buffer,
            ledger=ledger,
            expected_reasons={"token_mac_mismatch"},
        )

        ledger, store, request, buffer, token, _ = prepared("stale-epoch")
        store.advance(next_epoch_snapshot(bundle.snapshot))
        result = store.compare_and_commit(token, bundle.prices, request, buffer)
        record(
            dataset=dataset,
            case="stale_epoch",
            expected_release=False,
            result=result,
            buffer=buffer,
            ledger=ledger,
            expected_reasons={"stale_epoch"},
        )

        clock = _MutableClock(1_000)
        ledger, store, request, buffer, token, _ = prepared(
            "expired-token", clock=clock, ttl_ns=10
        )
        clock.value = 1_011
        result = store.compare_and_commit(token, bundle.prices, request, buffer)
        record(
            dataset=dataset,
            case="expired_token",
            expected_release=False,
            result=result,
            buffer=buffer,
            ledger=ledger,
            expected_reasons={"token_expired"},
        )

        ledger, store, request, buffer, token, _ = prepared("secret-rotation")
        store.restart_with_new_secret()
        result = store.compare_and_commit(token, bundle.prices, request, buffer)
        record(
            dataset=dataset,
            case="secret_rotation",
            expected_release=False,
            result=result,
            buffer=buffer,
            ledger=ledger,
            expected_reasons={"token_mac_mismatch"},
        )

        # Safety-critical sparse arrays are rehashed on the commit path.
        mutable_matrix = bundle.snapshot.matrix.copy()
        mutable_bounds = bundle.snapshot.bounds.copy()
        mutable_snapshot = core.ConstraintSnapshot(
            asset_ids=bundle.snapshot.asset_ids,
            matrix=mutable_matrix,
            bounds=mutable_bounds,
            row_metadata=bundle.snapshot.row_metadata.copy(deep=True),
            epoch=bundle.snapshot.epoch,
            constraint_digest=bundle.snapshot.constraint_digest,
            asset_digest=bundle.snapshot.asset_digest,
        )
        mutation_path = dataset_dir / "snapshot-mutation.sqlite"
        if mutation_path.exists():
            mutation_path.unlink()
        ledger = DurableReleaseLedger(mutation_path)
        store = RequestBoundSnapshotStore(
            mutable_snapshot,
            ledger=ledger,
            asset_query_digests=bundle.query_digests,
            secret=secrets.token_bytes(32),
            verifier_tolerance=tolerance,
        )
        request = _request_for_row(
            row_a,
            price_map,
            request_id=f"{_safe_slug(dataset)}-snapshot-mutation",
        )
        buffer = BoundAnswerBuffer(request=request, payload=answer_a)
        token = store.issue_token(
            mutable_snapshot, bundle.prices, request, buffer
        )
        mutable_matrix.data[0] += 0.25
        result = store.compare_and_commit(
            token, bundle.prices, request, buffer
        )
        record(
            dataset=dataset,
            case="snapshot_mutation_after_issue",
            expected_release=False,
            result=result,
            buffer=buffer,
            ledger=ledger,
            expected_reasons={"snapshot_integrity_failure"},
        )

        ledger, store, request, buffer, token, _ = prepared("token-replay")
        first = store.compare_and_commit(token, bundle.prices, request, buffer)
        replay = store.compare_and_commit(token, bundle.prices, request, buffer)
        fault_rows.append(
            {
                "dataset": dataset,
                "case": "token_replay",
                "phase": "second_commit",
                "expected_release": False,
                "released": replay.released,
                "durable": replay.durable,
                "reason": replay.reason,
                "buffer_released": buffer.released,
                "durable_release_rows": len(ledger.release_rows()),
                "passed": bool(
                    first.released
                    and not replay.released
                    and replay.reason == "token_replay"
                    and len(ledger.release_rows()) == 1
                ),
            }
        )

        # A request identifier is an idempotency key even when two independently
        # valid tokens and buffers exist for it.
        ledger_path = dataset_dir / "request-replay.sqlite"
        if ledger_path.exists():
            ledger_path.unlink()
        ledger = DurableReleaseLedger(ledger_path)
        shared_secret = secrets.token_bytes(32)
        store = RequestBoundSnapshotStore(
            bundle.snapshot,
            ledger=ledger,
            asset_query_digests=bundle.query_digests,
            secret=shared_secret,
            verifier_tolerance=tolerance,
        )
        request = _request_for_row(
            row_a,
            price_map,
            request_id=f"{_safe_slug(dataset)}-request-replay",
        )
        first_buffer = BoundAnswerBuffer(request=request, payload=answer_a)
        second_buffer = BoundAnswerBuffer(request=request, payload=answer_a)
        first_token = store.issue_token(
            bundle.snapshot, bundle.prices, request, first_buffer
        )
        second_token = store.issue_token(
            bundle.snapshot, bundle.prices, request, second_buffer
        )
        first_result = store.compare_and_commit(
            first_token, bundle.prices, request, first_buffer
        )
        second_result = store.compare_and_commit(
            second_token, bundle.prices, request, second_buffer
        )
        fault_rows.append(
            {
                "dataset": dataset,
                "case": "request_replay_distinct_token",
                "phase": "second_commit",
                "expected_release": False,
                "released": second_result.released,
                "durable": second_result.durable,
                "reason": second_result.reason,
                "buffer_released": second_buffer.released,
                "durable_release_rows": len(ledger.release_rows()),
                "passed": bool(
                    first_result.released
                    and not second_result.released
                    and second_result.reason == "request_replay"
                    and len(ledger.release_rows()) == 1
                ),
            }
        )

        ledger, store, request, buffer, token, shared_secret = prepared(
            "durable-replay"
        )
        first = store.compare_and_commit(token, bundle.prices, request, buffer)
        ledger.close_thread_connection()
        reopened = DurableReleaseLedger(ledger.path)
        restarted_store = RequestBoundSnapshotStore(
            bundle.snapshot,
            ledger=reopened,
            asset_query_digests=bundle.query_digests,
            secret=shared_secret,
            verifier_tolerance=tolerance,
        )
        replay_buffer = BoundAnswerBuffer(
            request=request,
            payload=answer_a,
            buffer_id=buffer.buffer_id,
        )
        replay = restarted_store.compare_and_commit(
            token, bundle.prices, request, replay_buffer
        )
        fault_rows.append(
            {
                "dataset": dataset,
                "case": "durable_replay_after_restart",
                "phase": "restart_commit",
                "expected_release": False,
                "released": replay.released,
                "durable": replay.durable,
                "reason": replay.reason,
                "buffer_released": replay_buffer.released,
                "durable_release_rows": len(reopened.release_rows()),
                "passed": bool(
                    first.released
                    and not replay.released
                    and replay.reason == "token_replay"
                    and len(reopened.release_rows()) == 1
                ),
            }
        )

        # Invalid registry and quote combinations must fail before token issuance.
        for label, invalid_request in [
            (
                "unregistered_query_at_issue",
                replace(
                    _request_for_row(
                        row_a,
                        price_map,
                        request_id=f"{_safe_slug(dataset)}-invalid-query",
                    ),
                    query_digest=exact_sql_digest("SELECT 999 AS invalid"),
                ),
            ),
            (
                "wrong_quote_at_issue",
                replace(
                    _request_for_row(
                        row_a,
                        price_map,
                        request_id=f"{_safe_slug(dataset)}-invalid-quote",
                    ),
                    quoted_price_hex=float(
                        np.nextafter(price_map[str(row_a["asset_id"])], math.inf)
                    ).hex(),
                ),
            ),
        ]:
            ledger_path = dataset_dir / f"issue-{_safe_slug(label)}.sqlite"
            if ledger_path.exists():
                ledger_path.unlink()
            ledger = DurableReleaseLedger(ledger_path)
            store = RequestBoundSnapshotStore(
                bundle.snapshot,
                ledger=ledger,
                asset_query_digests=bundle.query_digests,
                secret=secrets.token_bytes(32),
                verifier_tolerance=tolerance,
            )
            invalid_buffer = BoundAnswerBuffer(request=invalid_request, payload=answer_a)
            try:
                store.issue_token(
                    bundle.snapshot,
                    bundle.prices,
                    invalid_request,
                    invalid_buffer,
                )
                rejected = False
                reason = "unexpected_token_issuance"
            except ValueError as exc:
                rejected = True
                reason = str(exc)
            fault_rows.append(
                {
                    "dataset": dataset,
                    "case": label,
                    "phase": "issuance",
                    "expected_release": False,
                    "released": False,
                    "durable": False,
                    "reason": reason,
                    "buffer_released": invalid_buffer.released,
                    "durable_release_rows": len(ledger.release_rows()),
                    "passed": bool(rejected and len(ledger.release_rows()) == 0),
                }
            )

        # Cross-service races exercise SQLite uniqueness and serialization, not
        # only one store's in-process RLock.
        schedules = {
            "duplicate_token": "token_replay",
            "distinct_tokens_same_request": "request_replay",
        }
        for schedule, expected_loser_reason in schedules.items():
            ledger_path = dataset_dir / f"race-{schedule}.sqlite"
            if ledger_path.exists():
                ledger_path.unlink()
            ledger_a = DurableReleaseLedger(ledger_path)
            ledger_b = DurableReleaseLedger(ledger_path)
            shared_secret = secrets.token_bytes(32)
            store_a = RequestBoundSnapshotStore(
                bundle.snapshot,
                ledger=ledger_a,
                asset_query_digests=bundle.query_digests,
                secret=shared_secret,
                verifier_tolerance=tolerance,
            )
            store_b = RequestBoundSnapshotStore(
                bundle.snapshot,
                ledger=ledger_b,
                asset_query_digests=bundle.query_digests,
                secret=shared_secret,
                verifier_tolerance=tolerance,
            )
            for trial in range(int(concurrency_trials)):
                race_request = _request_for_row(
                    row_a,
                    price_map,
                    request_id=(
                        f"{_safe_slug(dataset)}-{schedule}-race-{trial}"
                    ),
                )
                if schedule == "duplicate_token":
                    buffer_id = (
                        f"{_safe_slug(dataset)}-duplicate-token-buffer-{trial}"
                    )
                    buffer_a = BoundAnswerBuffer(
                        request=race_request,
                        payload=answer_a,
                        buffer_id=buffer_id,
                    )
                    buffer_b = BoundAnswerBuffer(
                        request=race_request,
                        payload=answer_a,
                        buffer_id=buffer_id,
                    )
                    token_a = store_a.issue_token(
                        bundle.snapshot, bundle.prices, race_request, buffer_a
                    )
                    token_b = token_a
                else:
                    buffer_a = BoundAnswerBuffer(
                        request=race_request, payload=answer_a
                    )
                    buffer_b = BoundAnswerBuffer(
                        request=race_request, payload=answer_a
                    )
                    token_a = store_a.issue_token(
                        bundle.snapshot, bundle.prices, race_request, buffer_a
                    )
                    token_b = store_b.issue_token(
                        bundle.snapshot, bundle.prices, race_request, buffer_b
                    )

                barrier = threading.Barrier(3)
                result_holder: list[Any] = []
                result_lock = threading.Lock()

                def committer(
                    store,
                    candidate_buffer,
                    candidate_token,
                    ledger_instance,
                ) -> None:
                    barrier.wait(timeout=10)
                    outcome = store.compare_and_commit(
                        candidate_token,
                        bundle.prices,
                        race_request,
                        candidate_buffer,
                    )
                    with result_lock:
                        result_holder.append(outcome)
                    ledger_instance.close_thread_connection()

                threads = [
                    threading.Thread(
                        target=committer,
                        args=(store_a, buffer_a, token_a, ledger_a),
                    ),
                    threading.Thread(
                        target=committer,
                        args=(store_b, buffer_b, token_b, ledger_b),
                    ),
                ]
                for thread in threads:
                    thread.start()
                barrier.wait(timeout=10)
                for thread in threads:
                    thread.join(timeout=10)
                releases = sum(bool(outcome.released) for outcome in result_holder)
                ledger_rows = len(ledger_a.release_rows())
                reasons = sorted(outcome.reason for outcome in result_holder)
                passed = bool(
                    len(result_holder) == 2
                    and releases == 1
                    and ledger_rows == trial + 1
                    and reasons == ["released", expected_loser_reason]
                )
                concurrency_rows.append(
                    {
                        "dataset": dataset,
                        "schedule": schedule,
                        "trial": trial,
                        "service_instances": 2,
                        "completed_results": len(result_holder),
                        "released_results": releases,
                        "durable_release_rows": ledger_rows,
                        "expected_durable_release_rows": trial + 1,
                        "reasons": json.dumps(reasons),
                        "passed": passed,
                    }
                )

    return (
        pd.DataFrame(fault_rows),
        pd.DataFrame(concurrency_rows),
        pd.DataFrame(delivery_rows),
        pd.DataFrame(durable_rows),
    )


def _candidate_digest(asset_ids: Sequence[str], candidate: np.ndarray) -> str:
    return core.price_vector_digest(asset_ids, np.asarray(candidate, dtype=np.float64))


def _greedy_disjoint_rows(
    snapshot: core.ConstraintSnapshot,
    center: np.ndarray,
    relation_family: str,
    *,
    tolerance: float,
    budget_fraction: float,
) -> tuple[list[list[tuple[str, int, tuple[int, ...]]]], np.ndarray, np.ndarray]:
    scale = max(float(np.median(np.abs(center))), tolerance)
    widths = float(budget_fraction) * np.maximum(np.abs(center), scale)
    lower = np.maximum(0.0, center - widths)
    upper = center + widths
    attackable: list[tuple[str, int, tuple[int, ...]]] = []
    for row, metadata in snapshot.row_metadata.iterrows():
        if str(metadata["constraint_type"]) != relation_family:
            continue
        start, end = snapshot.matrix.indptr[row], snapshot.matrix.indptr[row + 1]
        columns = snapshot.matrix.indices[start:end]
        coefficients = snapshot.matrix.data[start:end]
        extremal = np.where(coefficients >= 0.0, upper[columns], lower[columns])
        residual = float(np.dot(coefficients, extremal) - snapshot.bounds[row])
        if residual > tolerance:
            attackable.append(
                (str(metadata["row_id"]), int(row), tuple(int(column) for column in columns))
            )
    attackable.sort(key=lambda record: record[0])
    pools: list[list[tuple[str, int, tuple[int, ...]]]] = []
    for parity in [0, 1]:
        used: set[int] = set()
        selected = []
        for index, record in enumerate(attackable):
            if index % 2 != parity:
                continue
            if used.isdisjoint(record[2]):
                selected.append(record)
                used.update(record[2])
        pools.append(selected)
    return pools, lower, upper


def build_dense_candidate_catalog(
    bundles: Mapping[str, DatasetBundle],
    *,
    tolerance: float,
    target_violation_counts: Sequence[int] = (8, 192),
    budget_fraction: float = 0.50,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Create deterministic multi-row attacks with genuinely dense active sets."""

    cases: list[dict[str, Any]] = []
    public_rows: list[dict[str, Any]] = []
    for dataset, bundle in bundles.items():
        for family in RELATION_FAMILIES:
            center, center_policy = relation_family_attack_center(
                bundle.snapshot,
                bundle.prices,
                family,
                tolerance=tolerance,
            )
            pools, lower, upper = _greedy_disjoint_rows(
                bundle.snapshot,
                center,
                family,
                tolerance=tolerance,
                budget_fraction=budget_fraction,
            )
            for pool_index, target_count in enumerate(target_violation_counts):
                pool = pools[pool_index % len(pools)]
                search_limit = min(96, len(pool))
                if search_limit < 2:
                    raise RuntimeError(
                        f"insufficient disjoint attack rows for {dataset}/{family}"
                    )
                candidates = []
                for selected_count in range(1, search_limit + 1):
                    candidate = center.copy()
                    chosen = pool[:selected_count]
                    for _, row, _ in chosen:
                        start, end = (
                            bundle.snapshot.matrix.indptr[row],
                            bundle.snapshot.matrix.indptr[row + 1],
                        )
                        columns = bundle.snapshot.matrix.indices[start:end]
                        coefficients = bundle.snapshot.matrix.data[start:end]
                        candidate[columns] = np.where(
                            coefficients >= 0.0,
                            upper[columns],
                            lower[columns],
                        )
                    verification = core.independent_verify(
                        bundle.snapshot, candidate, tolerance=tolerance
                    )
                    candidates.append(
                        (
                            abs(math.log1p(verification.violations) - math.log1p(target_count)),
                            selected_count,
                            verification,
                            candidate,
                            chosen,
                        )
                    )
                _, selected_count, verification, candidate, chosen = min(
                    candidates, key=lambda record: (record[0], record[1])
                )
                case_id = hashlib.sha256(
                    (
                        f"{dataset}|{family}|{pool_index}|{target_count}|"
                        f"{_candidate_digest(bundle.snapshot.asset_ids, candidate)}"
                    ).encode("utf-8")
                ).hexdigest()[:24]
                case = {
                    "dataset": dataset,
                    "relation_family": family,
                    "case_id": case_id,
                    "target_violation_count": int(target_count),
                    "selected_disjoint_rows": int(selected_count),
                    "selected_row_ids": [record[0] for record in chosen],
                    "violations_before": int(verification.violations),
                    "max_residual_before": float(verification.max_residual),
                    "candidate_digest": _candidate_digest(
                        bundle.snapshot.asset_ids, candidate
                    ),
                    "center_policy": center_policy,
                    "budget_fraction": float(budget_fraction),
                    "_candidate": candidate,
                    "_snapshot": bundle.snapshot,
                }
                cases.append(case)
                public_rows.append(
                    {
                        key: (
                            json.dumps(value)
                            if key == "selected_row_ids"
                            else value
                        )
                        for key, value in case.items()
                        if not key.startswith("_")
                    }
                )
    return cases, pd.DataFrame(public_rows)


def run_binding_budget_ablation(
    cases: Sequence[dict[str, Any]],
    *,
    budgets: Sequence[int],
    repetitions: int,
    warmups: int,
    tolerance: float,
    max_sweeps: int,
    step: float,
    osqp_max_iter: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    variants = [("active_only", False), ("fail_safe", True)]
    for case_index, case in enumerate(cases):
        snapshot = case["_snapshot"]
        candidate = case["_candidate"]
        precheck = core.independent_verify(snapshot, candidate, tolerance=tolerance)
        if precheck.passed or precheck.violations != case["violations_before"]:
            raise RuntimeError(f"dense candidate changed before ablation: {case['case_id']}")
        for budget in budgets:
            if isinstance(budget, bool) or int(budget) != budget or int(budget) < 1:
                raise ValueError("active-row budgets must be positive integers")
            options = {
                "max_sweeps": int(max_sweeps),
                "active_row_budget": int(budget),
                "step": float(step),
                "osqp_max_iter": int(osqp_max_iter),
            }
            for _ in range(int(warmups)):
                core.run_enforcement_method(
                    snapshot,
                    candidate,
                    method="arbiter_active",
                    tolerance=tolerance,
                    allow_fallback=False,
                    **options,
                )
                run_tiered_fail_safe_repair(
                    snapshot,
                    candidate,
                    tolerance=tolerance,
                    **options,
                )
            for repetition in range(int(repetitions)):
                rotation = (case_index + int(budget) + repetition) % len(variants)
                ordered_variants = variants[rotation:] + variants[:rotation]
                for execution_order, (variant, allow_fallback) in enumerate(
                    ordered_variants
                ):
                    if allow_fallback:
                        result = run_tiered_fail_safe_repair(
                            snapshot,
                            candidate,
                            tolerance=tolerance,
                            **options,
                        )
                    else:
                        result = core.run_enforcement_method(
                            snapshot,
                            candidate,
                            method="arbiter_active",
                            tolerance=tolerance,
                            allow_fallback=False,
                            **options,
                        )
                    oracle = core.independent_verify(
                        snapshot, result.prices, tolerance=tolerance
                    )
                    rows.append(
                        {
                            "dataset": case["dataset"],
                            "relation_family": case["relation_family"],
                            "case_id": case["case_id"],
                            "target_violation_count": case["target_violation_count"],
                            "violations_before": precheck.violations,
                            "budget_binding": bool(precheck.violations > int(budget)),
                            "active_row_budget": int(budget),
                            "max_sweeps": int(max_sweeps),
                            "variant": variant,
                            "repetition": repetition,
                            "execution_order": execution_order,
                            "runtime_ms": result.runtime_ms,
                            "fallback_ms": result.fallback_ms,
                            "active_path_ms": result.runtime_ms - result.fallback_ms,
                            "fallback_used": result.fallback_used,
                            "fallback_stage": getattr(
                                result, "fallback_stage", "none"
                            ),
                            "rejected": result.rejected,
                            "verified": result.verification.passed,
                            "oracle_verified": oracle.passed,
                            "violations_after": oracle.violations,
                            "max_residual_after": oracle.max_residual,
                            "unsafe_release": bool(
                                not result.rejected and not oracle.passed
                            ),
                            "sweeps": result.sweeps,
                            "projected_rows": result.projected_rows,
                            "l1_adjustment": result.l1_adjustment,
                            "solver": result.solver,
                            "candidate_digest": case["candidate_digest"],
                            "constraint_digest": snapshot.constraint_digest,
                        }
                    )
    raw = pd.DataFrame(rows)
    summary = (
        raw.groupby(
            [
                "dataset",
                "relation_family",
                "case_id",
                "violations_before",
                "active_row_budget",
                "budget_binding",
                "variant",
            ],
            as_index=False,
        )
        .agg(
            timing_repetitions=("runtime_ms", "size"),
            runtime_p50_ms=("runtime_ms", "median"),
            runtime_p95_ms=("runtime_ms", lambda values: float(np.quantile(values, 0.95))),
            active_path_p50_ms=("active_path_ms", "median"),
            fallback_rate=("fallback_used", "mean"),
            rejection_rate=("rejected", "mean"),
            verified_rate=("oracle_verified", "mean"),
            unsafe_release_rate=("unsafe_release", "mean"),
            projected_rows_median=("projected_rows", "median"),
            sweeps_median=("sweeps", "median"),
            l1_adjustment_median=("l1_adjustment", "median"),
        )
        .sort_values(
            ["dataset", "relation_family", "case_id", "active_row_budget", "variant"]
        )
        .reset_index(drop=True)
    )
    return raw, summary


def _target_margin_candidate(
    snapshot: core.ConstraintSnapshot,
    center: np.ndarray,
    family: str,
    margin: float,
) -> tuple[np.ndarray, int, str, float]:
    family_rows = snapshot.row_metadata[
        snapshot.row_metadata["constraint_type"].astype(str).eq(family)
    ].sort_values("row_id")
    if family_rows.empty:
        raise RuntimeError(f"no {family} row for sensitivity campaign")
    for row in family_rows.index:
        start, end = snapshot.matrix.indptr[row], snapshot.matrix.indptr[row + 1]
        columns = snapshot.matrix.indices[start:end]
        coefficients = snapshot.matrix.data[start:end]
        positive = np.flatnonzero(coefficients > 0.0)
        if not positive.size:
            continue
        current_residual = float(
            np.dot(coefficients, center[columns]) - snapshot.bounds[row]
        )
        coefficient_position = int(positive[0])
        coefficient = float(coefficients[coefficient_position])
        delta = (float(margin) - current_residual) / coefficient
        if delta <= 0.0:
            continue
        candidate = center.copy()
        candidate[int(columns[coefficient_position])] += delta
        target_residual = float(
            np.dot(coefficients, candidate[columns]) - snapshot.bounds[row]
        )
        if target_residual <= 0.0:
            continue
        return candidate, int(row), str(family_rows.loc[row, "row_id"]), target_residual
    raise RuntimeError(f"could not construct a positive-margin {family} candidate")


def run_tolerance_scale_sensitivity(
    bundles: Mapping[str, DatasetBundle],
    *,
    scale_factors: Sequence[float],
    relative_tolerances: Sequence[float],
    margin_ratios: Sequence[float],
    fixed_absolute_tolerance: float,
    max_sweeps: int,
    active_row_budget: int,
    step: float,
    osqp_max_iter: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    fixed_rows: list[dict[str, Any]] = []
    for dataset, bundle in bundles.items():
        if not np.allclose(bundle.snapshot.bounds, 0.0, atol=0.0, rtol=0.0):
            raise RuntimeError(
                "scale sensitivity requires the audited homogeneous zero-bound snapshot"
            )
        for family in RELATION_FAMILIES:
            center, center_policy = relation_family_attack_center(
                bundle.snapshot,
                bundle.prices,
                family,
                tolerance=1e-12,
            )
            positive = np.abs(center[np.isfinite(center) & (center > 0.0)])
            if not positive.size:
                raise RuntimeError(f"no positive prices for {dataset}/{family}")
            raw_reference = float(np.quantile(positive, 0.95))
            if not math.isfinite(raw_reference) or raw_reference <= 0.0:
                raise RuntimeError("invalid p95 normalization reference")
            normalized_center = center / raw_reference
            base_check = core.independent_verify(
                bundle.snapshot, normalized_center, tolerance=1e-12
            )
            if not base_check.passed:
                raise RuntimeError(f"normalization broke feasibility for {dataset}/{family}")

            for scale_factor in scale_factors:
                reference_unit = float(scale_factor)
                scaled_center = normalized_center * reference_unit
                for margin_ratio in margin_ratios:
                    margin = float(margin_ratio) * reference_unit
                    candidate, target_row, target_row_id, target_residual = _target_margin_candidate(
                        bundle.snapshot,
                        scaled_center,
                        family,
                        margin,
                    )
                    candidate_digest = _candidate_digest(
                        bundle.snapshot.asset_ids, candidate
                    )
                    fixed_check = core.independent_verify(
                        bundle.snapshot,
                        candidate,
                        tolerance=fixed_absolute_tolerance,
                    )
                    fixed_rows.append(
                        {
                            "dataset": dataset,
                            "relation_family": family,
                            "center_policy": center_policy,
                            "normalization": "p95_positive_price_equals_one",
                            "raw_p95_reference": raw_reference,
                            "price_scale_factor": reference_unit,
                            "margin_ratio": float(margin_ratio),
                            "target_row": target_row,
                            "target_row_id": target_row_id,
                            "target_residual": target_residual,
                            "fixed_absolute_tolerance": float(fixed_absolute_tolerance),
                            "violations": fixed_check.violations,
                            "accepted": fixed_check.passed,
                            "candidate_digest": candidate_digest,
                        }
                    )
                    for relative_tolerance in relative_tolerances:
                        absolute_tolerance = (
                            float(relative_tolerance) * reference_unit
                        )
                        before = core.independent_verify(
                            bundle.snapshot,
                            candidate,
                            tolerance=absolute_tolerance,
                        )
                        normalized_candidate = candidate / reference_unit
                        normalized_before = core.independent_verify(
                            bundle.snapshot,
                            normalized_candidate,
                            tolerance=float(relative_tolerance),
                        )
                        if (
                            before.passed != normalized_before.passed
                            or before.violations != normalized_before.violations
                        ):
                            raise RuntimeError(
                                "raw and normalized pre-verification decisions diverged"
                            )
                        # Repair in declared p95-normalized price units. This
                        # removes arbitrary currency magnitude from stopping
                        # logic; the final raw-unit vector is still independently
                        # verified under the signed absolute tolerance.
                        result = core.run_enforcement_method(
                            bundle.snapshot,
                            normalized_candidate,
                            method="arbiter_active",
                            tolerance=float(relative_tolerance),
                            max_sweeps=max_sweeps,
                            active_row_budget=active_row_budget,
                            step=step,
                            allow_fallback=True,
                            osqp_max_iter=osqp_max_iter,
                        )
                        repaired_raw = result.prices * reference_unit
                        oracle = core.independent_verify(
                            bundle.snapshot,
                            repaired_raw,
                            tolerance=absolute_tolerance,
                        )
                        rows.append(
                            {
                                "dataset": dataset,
                                "relation_family": family,
                                "center_policy": center_policy,
                                "normalization": "p95_positive_price_equals_one",
                                "repair_space": "p95_normalized",
                                "raw_p95_reference": raw_reference,
                                "price_scale_factor": reference_unit,
                                "reference_unit": reference_unit,
                                "margin_ratio": float(margin_ratio),
                                "relative_tolerance": float(relative_tolerance),
                                "absolute_tolerance": absolute_tolerance,
                                "target_row": target_row,
                                "target_row_id": target_row_id,
                                "target_residual": target_residual,
                                "target_residual_ratio": target_residual / reference_unit,
                                "violations_before": before.violations,
                                "accepted_without_repair": before.passed,
                                "verified_after": oracle.passed,
                                "rejected": result.rejected,
                                "fallback_used": result.fallback_used,
                                "unsafe_release": bool(
                                    not result.rejected and not oracle.passed
                                ),
                                "violations_after": oracle.violations,
                                "max_residual_after": oracle.max_residual,
                                "runtime_ms": result.runtime_ms,
                                "l1_adjustment": float(
                                    np.abs(repaired_raw - candidate).sum()
                                ),
                                "normalized_l1_adjustment": result.l1_adjustment,
                                "candidate_digest": candidate_digest,
                                "constraint_digest": bundle.snapshot.constraint_digest,
                            }
                        )

    raw = pd.DataFrame(rows)
    fixed = pd.DataFrame(fixed_rows)
    invariance_rows = []
    group_keys = [
        "dataset",
        "relation_family",
        "margin_ratio",
        "relative_tolerance",
    ]
    for keys, group in raw.groupby(group_keys, sort=True):
        violations_invariant = group["violations_before"].nunique() == 1
        acceptance_invariant = group["accepted_without_repair"].nunique() == 1
        release_invariant = group["rejected"].nunique() == 1
        distortion = group["normalized_l1_adjustment"].to_numpy(float)
        distortion_span = float(distortion.max() - distortion.min())
        distortion_scale = max(float(np.abs(distortion).max()), 1e-15)
        distortion_relative_span = distortion_span / distortion_scale
        # A percentage is unstable when the adjustment itself is near zero.
        # Judge scale invariance in normalized price units, with a bound tied to
        # the configured verifier tolerance, while retaining the relative span
        # as a diagnostic column.
        distortion_span_limit = max(1e-9, 5.0 * float(keys[3]))
        invariance_rows.append(
            {
                **dict(zip(group_keys, keys, strict=True)),
                "scale_points": int(group["price_scale_factor"].nunique()),
                "violations_invariant": violations_invariant,
                "acceptance_invariant": acceptance_invariant,
                "release_invariant": release_invariant,
                "normalized_distortion_span": distortion_span,
                "normalized_distortion_relative_span": distortion_relative_span,
                "normalized_distortion_span_limit": distortion_span_limit,
                "passed": bool(
                    violations_invariant
                    and acceptance_invariant
                    and release_invariant
                    and distortion_span <= distortion_span_limit
                    and group["verified_after"].all()
                    and not group["unsafe_release"].any()
                ),
            }
        )
    invariance = pd.DataFrame(invariance_rows)

    fixed_drift_rows = []
    fixed_keys = ["dataset", "relation_family", "margin_ratio"]
    for keys, group in fixed.groupby(fixed_keys, sort=True):
        fixed_drift_rows.append(
            {
                **dict(zip(fixed_keys, keys, strict=True)),
                "scale_points": int(group["price_scale_factor"].nunique()),
                "distinct_acceptance_decisions": int(group["accepted"].nunique()),
                "decision_drift_across_scale": bool(group["accepted"].nunique() > 1),
            }
        )
    fixed_drift = pd.DataFrame(fixed_drift_rows)
    return raw, invariance, fixed, fixed_drift


def run_protocol_overhead_benchmark(
    bundles: Mapping[str, DatasetBundle],
    *,
    answer_executor: Callable[[str, str], Any],
    work_dir: Path,
    tolerance: float,
    repetitions: int,
    warmups: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate answer hashing/staging, authorization, commit, and delivery costs."""

    work_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for dataset, bundle in bundles.items():
        row_a, _ = _protocol_assets(bundle)
        answer = answer_executor(dataset, str(row_a["sql"]))
        price_map = dict(zip(bundle.snapshot.asset_ids, bundle.prices, strict=True))
        ledger_path = work_dir / f"overhead-{_safe_slug(dataset)}.sqlite"
        if ledger_path.exists():
            ledger_path.unlink()
        ledger = DurableReleaseLedger(ledger_path)
        store = RequestBoundSnapshotStore(
            bundle.snapshot,
            ledger=ledger,
            asset_query_digests=bundle.query_digests,
            secret=secrets.token_bytes(32),
            verifier_tolerance=tolerance,
        )

        def one(repetition: int, measured: bool) -> None:
            request = _request_for_row(
                row_a,
                price_map,
                request_id=f"overhead-{_safe_slug(dataset)}-{repetition}-{'m' if measured else 'w'}",
            )
            started = time.perf_counter_ns()
            buffer = BoundAnswerBuffer(request=request, payload=answer)
            buffer_ms = (time.perf_counter_ns() - started) / 1e6

            started = time.perf_counter_ns()
            ledger.stage_buffer(request, buffer)
            stage_ms = (time.perf_counter_ns() - started) / 1e6

            started = time.perf_counter_ns()
            token = store.issue_token(
                bundle.snapshot, bundle.prices, request, buffer
            )
            issue_ms = (time.perf_counter_ns() - started) / 1e6

            started = time.perf_counter_ns()
            commit = store.compare_and_commit(
                token, bundle.prices, request, buffer
            )
            commit_ms = (time.perf_counter_ns() - started) / 1e6

            started = time.perf_counter_ns()
            delivered = ledger.released_payload(
                token_id=token.token_id, request=request
            )
            delivery_ms = (time.perf_counter_ns() - started) / 1e6
            if not commit.released or delivered != buffer.payload_bytes:
                raise RuntimeError("protocol overhead benchmark failed its release check")
            if measured:
                rows.append(
                    {
                        "dataset": dataset,
                        "repetition": repetition,
                        "payload_bytes": len(buffer.payload_bytes),
                        "buffer_digest_ms": buffer_ms,
                        "durable_stage_ms": stage_ms,
                        "token_issue_ms": issue_ms,
                        "durable_commit_ms": commit_ms,
                        "delivery_integrity_ms": delivery_ms,
                        "authorization_ms": issue_ms + commit_ms,
                        "control_without_delivery_ms": stage_ms + issue_ms + commit_ms,
                        "complete_buffer_to_delivery_ms": (
                            buffer_ms + stage_ms + issue_ms + commit_ms + delivery_ms
                        ),
                        "released": commit.released,
                        "durable": commit.durable,
                    }
                )

        for warmup in range(int(warmups)):
            one(-warmup - 1, False)
        for repetition in range(int(repetitions)):
            one(repetition, True)

    raw = pd.DataFrame(rows)
    summary = (
        raw.groupby("dataset", as_index=False)
        .agg(
            timing_repetitions=("repetition", "size"),
            payload_bytes=("payload_bytes", "first"),
            buffer_digest_p50_ms=("buffer_digest_ms", "median"),
            stage_p50_ms=("durable_stage_ms", "median"),
            issue_p50_ms=("token_issue_ms", "median"),
            commit_p50_ms=("durable_commit_ms", "median"),
            delivery_p50_ms=("delivery_integrity_ms", "median"),
            authorization_p50_ms=("authorization_ms", "median"),
            complete_p50_ms=("complete_buffer_to_delivery_ms", "median"),
            complete_p95_ms=(
                "complete_buffer_to_delivery_ms",
                lambda values: float(np.quantile(values, 0.95)),
            ),
            release_rate=("released", "mean"),
            durable_rate=("durable", "mean"),
        )
        .sort_values("dataset")
        .reset_index(drop=True)
    )
    return raw, summary


def hardware_environment_manifest(extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
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
    clock = time.get_clock_info("perf_counter")
    result = core.environment_manifest(extra=extra)
    result.update(
        {
            "closure_suite_version": CLOSURE_SUITE_VERSION,
            "cpu_model": cpu_model,
            "logical_cpu_count": os.cpu_count(),
            "memory_bytes": memory_bytes,
            "machine": platform.machine(),
            "perf_counter_resolution_seconds": clock.resolution,
            "perf_counter_monotonic": clock.monotonic,
            "sqlite_version": __import__("sqlite3").sqlite_version,
            "thread_environment": {
                name: os.environ.get(name, "unset")
                for name in [
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                ]
            },
        }
    )
    return result


def closure_quality_checks(
    *,
    input_audit: pd.DataFrame,
    protocol_faults: pd.DataFrame,
    concurrency: pd.DataFrame,
    delivery: pd.DataFrame,
    budget_raw: pd.DataFrame,
    tolerance_raw: pd.DataFrame,
    invariance: pd.DataFrame,
    fixed_drift: pd.DataFrame,
    overhead_raw: pd.DataFrame,
) -> pd.DataFrame:
    checks = [
        ("all audited-v4 input hashes and code hashes match", input_audit["passed"].all()),
        ("every protocol substitution and issuance fault passes", protocol_faults["passed"].all()),
        ("authorized and unauthorized deliveries behave correctly", delivery["passed"].all()),
        ("cross-service duplicate-token and same-request races allow exactly one release", concurrency["passed"].all()),
        ("every tested active-row budget is binding in at least one case", all(
            budget_raw.loc[budget_raw["active_row_budget"] == budget, "budget_binding"].any()
            for budget in sorted(budget_raw["active_row_budget"].unique())
        )),
        ("dense ablation contains violations above the former observed maximum", budget_raw["violations_before"].max() > 47),
        ("active-only ablation exercises rejection", budget_raw.loc[budget_raw["variant"] == "active_only", "rejected"].any()),
        ("fail-safe ablation exercises fallback", budget_raw.loc[budget_raw["variant"] == "fail_safe", "fallback_used"].any()),
        ("fail-safe ablation releases no unverified vector", not budget_raw.loc[budget_raw["variant"] == "fail_safe", "unsafe_release"].any()),
        ("scale-aware tolerance decisions and normalized distortion are invariant", invariance["passed"].all()),
        ("fixed absolute tolerance diagnostic exposes at least one scale-dependent decision", fixed_drift["decision_drift_across_scale"].any()),
        ("all scale-aware repaired outputs independently verify", tolerance_raw["verified_after"].all()),
        ("tolerance campaign records no unsafe release", not tolerance_raw["unsafe_release"].any()),
        ("protocol overhead trials all durably release", overhead_raw["released"].all() and overhead_raw["durable"].all()),
    ]
    return pd.DataFrame(
        [{"check": label, "passed": bool(passed)} for label, passed in checks]
    )
