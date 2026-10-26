"""Experiment helpers for the Arbiter Colab evidence suite."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
import threading
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from arbiter_core import (
    AnswerBuffer,
    ConstraintSnapshot,
    DEFAULT_TOLERANCE,
    ReleaseToken,
    SnapshotStore,
    build_snapshot,
    exact_box_row_attacks,
    extend_snapshot,
    independent_verify,
    replace_token,
    run_enforcement_method,
    snapshot_without_rows,
)
from arbiter_workload import QueryDescriptor


def stable_unit_interval(*parts: str) -> float:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], "big")
    return integer / float(2**64 - 1)


def add_structural_features(assets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in assets.itertuples(index=False):
        if str(row.oracle_descriptor):
            descriptor = QueryDescriptor.from_json(str(row.oracle_descriptor))
            interval_width = (
                float(descriptor.upper - descriptor.lower)
                if descriptor.predicate_kind == "numeric_interval"
                else 0.0
            )
            rows.append(
                {
                    "dataset": row.dataset,
                    "asset_id": row.asset_id,
                    "template_group": row.template_group,
                    "family": row.family,
                    "table": descriptor.table,
                    "mode": descriptor.mode,
                    "predicate_kind": descriptor.predicate_kind,
                    "aggregate": descriptor.aggregate or "none",
                    "output_count": len(descriptor.output_columns),
                    "category_count": len(descriptor.categories),
                    "interval_width": interval_width,
                    "sql_chars": len(str(row.sql)),
                    "stable_hash": stable_unit_interval(
                        str(row.dataset), str(row.asset_id)
                    ),
                    "expected_supported": bool(row.expected_supported),
                }
            )
        else:
            rows.append(
                {
                    "dataset": row.dataset,
                    "asset_id": row.asset_id,
                    "template_group": row.template_group,
                    "family": row.family,
                    "table": "unsupported",
                    "mode": "unsupported",
                    "predicate_kind": "unsupported",
                    "aggregate": "unsupported",
                    "output_count": 0,
                    "category_count": 0,
                    "interval_width": 0.0,
                    "sql_chars": len(str(row.sql)),
                    "stable_hash": stable_unit_interval(
                        str(row.dataset), str(row.asset_id)
                    ),
                    "expected_supported": False,
                }
            )
    features = pd.DataFrame(rows)
    if "estimated_rows" in assets.columns:
        features = features.merge(
            assets[["dataset", "asset_id", "estimated_rows"]],
            on=["dataset", "asset_id"],
            how="left",
        )
    else:
        features["estimated_rows"] = 0.0
    features["log_estimated_rows"] = np.log1p(
        features["estimated_rows"].fillna(0).clip(lower=0)
    )
    return features


def raw_controlled_policy(features: pd.DataFrame) -> np.ndarray:
    """A deterministic boundary-stress policy, not observed market prices.

    Price is linear in the partition extent and uses the same multiplier for a
    target and its disjoint components. Therefore generated decomposition rows
    are tight by construction, making 10/25/50% attack budgets meaningful rather
    than relying on accidental model error. Wider row-set projections cost more,
    so projection derivations remain feasible.
    """

    predicate_kind = features["predicate_kind"].to_numpy(str)
    interval = features["interval_width"].to_numpy(float)
    positive_intervals = interval[interval > 0]
    interval_scale = (
        float(positive_intervals.max())
        if positive_intervals.size
        else 1.0
    )
    category_count = features["category_count"].to_numpy(float)
    category_scale = max(float(category_count.max(initial=1.0)), 1.0)
    extent = np.where(
        predicate_kind == "numeric_interval",
        interval / max(interval_scale, 1e-12),
        np.where(
            predicate_kind == "category_set",
            category_count / category_scale,
            1.0,
        ),
    )
    extent = np.maximum(extent, 1e-9)
    output_multiplier = 1.0 + 0.12 * features["output_count"].to_numpy(float)
    aggregate_multiplier = np.where(
        features["mode"].to_numpy(str) == "additive_aggregate",
        1.25,
        1.0,
    )
    prices = extent * output_multiplier * aggregate_multiplier
    return prices.astype(np.float64)


def make_feasible_controlled_policy(
    snapshot: ConstraintSnapshot,
    ordered_features: pd.DataFrame,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> tuple[np.ndarray, dict[str, Any]]:
    raw = raw_controlled_policy(ordered_features)
    direct = independent_verify(snapshot, raw, tolerance=tolerance)
    if direct.passed:
        return raw, {
            "projected": False,
            "solver": "none",
            "l1_adjustment": 0.0,
            "verified_max_residual": direct.max_residual,
        }
    projected = run_enforcement_method(
        snapshot,
        raw,
        method="osqp_qp",
        tolerance=tolerance,
        osqp_max_iter=50_000,
    )
    if projected.rejected:
        projected = run_enforcement_method(
            snapshot,
            raw,
            method="full_cyclic",
            tolerance=tolerance,
            max_sweeps=500,
        )
    if projected.rejected:
        raise RuntimeError(
            "Could not construct a feasible controlled policy: " + projected.error
        )
    return projected.prices, {
        "projected": True,
        "solver": projected.solver,
        "l1_adjustment": projected.l1_adjustment,
        "verified_max_residual": projected.verification.max_residual,
    }


def relation_family_attack_center(
    snapshot: ConstraintSnapshot,
    base_center: np.ndarray,
    relation_type: str,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> tuple[np.ndarray, str]:
    """Return an explicitly labeled feasible center for one stress family.

    The additive controlled policy puts decomposition rows on their boundary.
    A constant positive vector puts projection/filter derivations on equality
    while remaining feasible for decomposition and nonnegativity rows. Using
    these centers makes 10/25/50% boxes genuine boundary stress tests; it is not
    presented as an observed seller-price distribution.
    """

    if relation_type == "decomposition":
        center = np.asarray(base_center, dtype=np.float64).copy()
        label = "additive_boundary_center"
    elif relation_type in {"projection_derivation", "filter_derivation"}:
        base = np.asarray(base_center, dtype=np.float64)
        positive = base[np.isfinite(base) & (base > 0)]
        level = float(np.median(positive)) if positive.size else 1.0
        level = max(level, 100.0 * float(tolerance))
        center = np.full(snapshot.assets, level, dtype=np.float64)
        label = "equal_price_derivation_boundary_center"
    else:
        raise ValueError(f"Unsupported attack relation family: {relation_type}")
    verification = independent_verify(snapshot, center, tolerance=tolerance)
    if not verification.passed:
        raise RuntimeError(
            f"Constructed {label} is infeasible for {relation_type}: "
            f"{verification.reason}"
        )
    return center, label


def _model_pipeline(
    *,
    model_name: str,
    seed: int,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            ("numeric", StandardScaler(), list(numeric_columns)),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(categorical_columns),
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    if model_name == "hist_gradient_boosting":
        model = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=31,
            l2_regularization=1e-3,
            random_state=int(seed),
        )
    elif model_name == "mlp":
        # Bounded activations avoid the severe held-out-group extrapolation
        # observed with ReLU on Adult (one fold produced prices two orders of
        # magnitude outside the label range). Early-stopped Adam scales much
        # better than LBFGS at the full 4,096-asset configuration. The MLP
        # remains a replaceable proposal, never an authority.
        model = MLPRegressor(
            hidden_layer_sizes=(32, 16),
            activation="tanh",
            solver="adam",
            alpha=1e-3,
            batch_size=128,
            learning_rate_init=5e-4,
            max_iter=1_000,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=40,
            random_state=int(seed),
        )
    elif model_name == "dummy_median":
        model = DummyRegressor(strategy="median")
    else:
        raise ValueError(f"Unknown proposal model: {model_name}")
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def grouped_oof_predictions(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    seeds: Sequence[int],
    model_names: Sequence[str] = (
        "hist_gradient_boosting",
        "mlp",
        "dummy_median",
    ),
    folds: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train grouped out-of-fold proposal models and retain every prediction."""

    data = features.merge(
        targets[["dataset", "asset_id", "target_price"]],
        on=["dataset", "asset_id"],
        how="inner",
        validate="one_to_one",
    )
    numeric_columns = [
        "output_count",
        "category_count",
        "interval_width",
        "sql_chars",
        "log_estimated_rows",
    ]
    categorical_columns = [
        "table",
        "mode",
        "predicate_kind",
        "aggregate",
        "family",
    ]
    metric_rows = []
    prediction_rows = []
    pooled_rows = []

    for dataset, frame in data.groupby("dataset", sort=False):
        frame = frame.reset_index(drop=True)
        groups = frame["template_group"].astype(str)
        unique_groups = groups.nunique()
        split_count = min(int(folds), int(unique_groups))
        if split_count < 2:
            raise ValueError(f"{dataset} has fewer than two template groups")
        splitter = GroupKFold(n_splits=split_count)
        x = frame[numeric_columns + categorical_columns]
        y = frame["target_price"].to_numpy(float)

        for model_name in model_names:
            for seed in seeds:
                oof = np.full(len(frame), np.nan, dtype=np.float64)
                for fold, (train_index, test_index) in enumerate(
                    splitter.split(x, y, groups)
                ):
                    pipeline = _model_pipeline(
                        model_name=model_name,
                        seed=int(seed),
                        numeric_columns=numeric_columns,
                        categorical_columns=categorical_columns,
                    )
                    started = time.perf_counter_ns()
                    pipeline.fit(x.iloc[train_index], y[train_index])
                    fit_ms = (time.perf_counter_ns() - started) / 1e6
                    started = time.perf_counter_ns()
                    predictions = pipeline.predict(x.iloc[test_index])
                    predict_ms = (time.perf_counter_ns() - started) / 1e6
                    oof[test_index] = predictions
                    metric_rows.append(
                        {
                            "dataset": dataset,
                            "model": model_name,
                            "seed": int(seed),
                            "fold": int(fold),
                            "train_assets": len(train_index),
                            "test_assets": len(test_index),
                            "held_out_groups": int(groups.iloc[test_index].nunique()),
                            "r2": float(r2_score(y[test_index], predictions)),
                            "mae": float(
                                mean_absolute_error(y[test_index], predictions)
                            ),
                            "fit_ms": float(fit_ms),
                            "predict_ms": float(predict_ms),
                        }
                    )
                if not np.isfinite(oof).all():
                    raise AssertionError("GroupKFold left missing OOF predictions")
                for asset_id, target, prediction in zip(
                    frame["asset_id"], y, oof, strict=True
                ):
                    prediction_rows.append(
                        {
                            "dataset": dataset,
                            "model": model_name,
                            "seed": int(seed),
                            "asset_id": asset_id,
                            "target_price": float(target),
                            "oof_prediction": float(prediction),
                        }
                    )
                pooled_rows.append(
                    {
                        "dataset": dataset,
                        "model": model_name,
                        "seed": int(seed),
                        "pooled_r2": float(r2_score(y, oof)),
                        "pooled_mae": float(mean_absolute_error(y, oof)),
                        "negative_predictions": int((oof < 0).sum()),
                    }
                )
    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(prediction_rows),
        pd.DataFrame(pooled_rows),
    )


def relation_subset(relations: pd.DataFrame, asset_ids: Iterable[str]) -> pd.DataFrame:
    allowed = set(str(item) for item in asset_ids)
    mask = []
    for row in relations.to_dict(orient="records"):
        if row["relation_type"] == "decomposition":
            referenced = {str(row["target_id"]), *json.loads(row["component_ids"])}
        else:
            referenced = {str(row["derived_id"]), str(row["source_id"])}
        mask.append(referenced <= allowed)
    return relations.loc[mask].reset_index(drop=True)


def _relation_asset_ids(record: Mapping[str, Any]) -> set[str]:
    if str(record["relation_type"]) == "decomposition":
        return {
            str(record["target_id"]),
            *[str(item) for item in json.loads(str(record["component_ids"]))],
        }
    return {str(record["derived_id"]), str(record["source_id"])}


def balanced_relation_closed_asset_ids(
    dataset_assets: pd.DataFrame,
    dataset_relations: pd.DataFrame,
    scale: int,
) -> list[str]:
    """Choose a deterministic scale subset that represents each relation family.

    Taking the first ``n`` generated assets delayed projection relationships in
    two datasets, making small-scale cells incomparable. This selector first
    closes at least one relation of every available family, then round-robins
    additional relation rows before filling remaining slots in workload order.
    """

    scale = int(scale)
    ordered = dataset_assets["asset_id"].astype(str).tolist()
    if scale < 1 or scale > len(ordered):
        raise ValueError(f"scale {scale} is outside 1..{len(ordered)}")
    relation_groups = {
        str(relation_type): group.sort_values("relation_id").to_dict(
            orient="records"
        )
        for relation_type, group in dataset_relations.groupby(
            "relation_type", sort=True
        )
    }
    if not relation_groups:
        raise ValueError("scale selection requires at least one relation")

    selected: set[str] = set()
    positions = {relation_type: 0 for relation_type in relation_groups}
    # First guarantee one closed row from each family.
    for relation_type in sorted(relation_groups):
        records = relation_groups[relation_type]
        chosen = None
        for position, record in enumerate(records):
            references = _relation_asset_ids(record)
            if len(selected | references) <= scale:
                chosen = (position, references)
                break
        if chosen is None:
            raise ValueError(
                f"scale {scale} cannot close a {relation_type} relation"
            )
        position, references = chosen
        selected.update(references)
        positions[relation_type] = position + 1

    # Add relation-closed assets round-robin until no further new references fit.
    while len(selected) < scale:
        made_progress = False
        for relation_type in sorted(relation_groups):
            records = relation_groups[relation_type]
            while positions[relation_type] < len(records):
                record = records[positions[relation_type]]
                positions[relation_type] += 1
                references = _relation_asset_ids(record)
                if references <= selected:
                    continue
                if len(selected | references) <= scale:
                    selected.update(references)
                    made_progress = True
                    break
            if len(selected) >= scale:
                break
        if not made_progress:
            break

    for asset_id in ordered:
        if len(selected) >= scale:
            break
        selected.add(asset_id)
    chosen_order = [asset_id for asset_id in ordered if asset_id in selected]
    if len(chosen_order) != scale:
        raise AssertionError("relation-closed scale selection returned wrong size")
    return chosen_order


def run_certified_attack_campaign(
    *,
    assets: pd.DataFrame,
    relations: pd.DataFrame,
    policy: pd.DataFrame,
    budgets: Sequence[float],
    requested_cases_per_relation_type: int,
    minimum_cases_per_relation_type: int,
    tolerance: float = DEFAULT_TOLERANCE,
    repair_options: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Attack unique rows in a balanced deterministic campaign.

    Exact row attacks have no random initialization. Treating model seeds as
    attack replicates would therefore manufacture pseudoreplication. Each
    dataset/budget/family cell uses distinct target rows, with the same number
    selected from every supported relation family.
    """

    options = dict(repair_options or {})
    result_rows = []
    token_rows = []
    delta_rows = []
    for dataset, dataset_assets in assets.groupby("dataset", sort=False):
        ordered_ids = dataset_assets["asset_id"].astype(str).tolist()
        dataset_relations = relations[relations["dataset"] == dataset]
        snapshot = build_snapshot(ordered_ids, dataset_relations, epoch=1)
        policy_map = (
            policy[policy["dataset"] == dataset]
            .set_index("asset_id")["target_price"]
            .to_dict()
        )
        center = np.asarray([policy_map[item] for item in ordered_ids], dtype=float)
        if not independent_verify(snapshot, center, tolerance=tolerance).passed:
            raise ValueError(f"Controlled policy for {dataset} is not feasible")

        for budget in budgets:
            relation_types = sorted(
                set(snapshot.row_metadata["constraint_type"])
                & {
                    "decomposition",
                    "projection_derivation",
                    "filter_derivation",
                }
            )
            requested = int(requested_cases_per_relation_type)
            minimum = int(minimum_cases_per_relation_type)
            if requested < minimum or minimum < 1:
                raise ValueError("invalid attack case-count configuration")
            per_type_centers = {
                relation_type: relation_family_attack_center(
                    snapshot,
                    center,
                    relation_type,
                    tolerance=tolerance,
                )
                for relation_type in relation_types
            }
            per_type_pools = {
                relation_type: exact_box_row_attacks(
                    snapshot,
                    per_type_centers[relation_type][0],
                    budget_fraction=float(budget),
                    limit=max(requested, minimum),
                    tolerance=tolerance,
                    include_types=[relation_type],
                )
                for relation_type in relation_types
            }
            missing = [name for name, pool in per_type_pools.items() if not pool]
            if missing:
                raise RuntimeError(
                    f"No attackable {missing} rows for {dataset} at budget {budget}"
                )
            cases_per_type = min(
                requested,
                min(len(pool) for pool in per_type_pools.values()),
            )
            if cases_per_type < minimum:
                available = {
                    relation_type: len(pool)
                    for relation_type, pool in per_type_pools.items()
                }
                raise RuntimeError(
                    f"Insufficient unique attack rows for {dataset}/{budget}: "
                    f"{available}; require at least {minimum} per family"
                )
            for relation_type in sorted(per_type_pools):
                attack_pool = per_type_pools[relation_type][:cases_per_type]
                for rank, attack in enumerate(attack_pool):
                    attack_center, attack_center_policy = per_type_centers[
                        relation_type
                    ]
                    candidate = attack["candidate"]
                    before = independent_verify(
                        snapshot, candidate, tolerance=tolerance
                    )
                    enforcement = run_enforcement_method(
                        snapshot,
                        candidate,
                        method="arbiter_active",
                        tolerance=tolerance,
                        **options,
                    )
                    released = not enforcement.rejected
                    unsafe = bool(
                        released and not enforcement.verification.passed
                    )
                    attack_id = hashlib.sha256(
                        (
                            f"{dataset}|{budget}|{relation_type}|{rank}|"
                            f"{attack['row_id']}"
                        ).encode("utf-8")
                    ).hexdigest()[:24]
                    result_rows.append(
                        {
                            "dataset": dataset,
                            "budget_fraction": float(budget),
                            "cases_per_relation_type": cases_per_type,
                            "attack_rank": int(rank),
                            "attack_id": attack_id,
                            "target_row_id": attack["row_id"],
                            "target_constraint_type": attack["constraint_type"],
                            "attack_center_policy": attack_center_policy,
                            "attack_center_digest": hashlib.sha256(
                                np.asarray(
                                    attack_center,
                                    dtype="<f8",
                                ).tobytes(order="C")
                            ).hexdigest(),
                            "certified_target_residual": attack[
                                "certified_target_residual"
                            ],
                            "violations_before": before.violations,
                            "max_residual_before": before.max_residual,
                            "violations_after": enforcement.verification.violations,
                            "max_residual_after": enforcement.verification.max_residual,
                            "released": released,
                            "rejected": enforcement.rejected,
                            "fallback_used": enforcement.fallback_used,
                            "unsafe_release": unsafe,
                            "repair_ms": enforcement.runtime_ms,
                            "fallback_ms": enforcement.fallback_ms,
                            "sweeps": enforcement.sweeps,
                            "projected_rows": enforcement.projected_rows,
                            "l1_adjustment": enforcement.l1_adjustment,
                            "constraint_digest": snapshot.constraint_digest,
                            "epoch": snapshot.epoch,
                        }
                    )
                    changed = np.flatnonzero(
                        np.abs(enforcement.prices - candidate) > 1e-14
                    )
                    for position in changed:
                        delta_rows.append(
                            {
                                "attack_id": attack_id,
                                "asset_id": ordered_ids[position],
                                "candidate_price": float(candidate[position]),
                                "released_price": float(
                                    enforcement.prices[position]
                                ),
                                "delta": float(
                                    enforcement.prices[position] - candidate[position]
                                ),
                            }
                        )
                    token_rows.append(
                        {
                            "attack_id": attack_id,
                            "constraint_digest": snapshot.constraint_digest,
                            "epoch": snapshot.epoch,
                            "verified": enforcement.verification.passed,
                            "verified_max_residual": (
                                enforcement.verification.max_residual
                            ),
                            "solver": enforcement.solver,
                        }
                    )
    return (
        pd.DataFrame(result_rows),
        pd.DataFrame(token_rows),
        pd.DataFrame(delta_rows),
    )


def run_omitted_relation_gap_campaign(
    *,
    assets: pd.DataFrame,
    relations: pd.DataFrame,
    policy: pd.DataFrame,
    budgets: Sequence[float],
    cases_per_relation_type: int,
    tolerance: float = DEFAULT_TOLERANCE,
    repair_options: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Measure what happens when one otherwise supported row is omitted.

    Each case uses an exact box attack against a named row in the complete
    snapshot, removes only that row, and runs the ordinary fail-closed gate on
    the incomplete snapshot.  A surviving gap is a released vector that passes
    every encoded row in the incomplete snapshot but violates the omitted row in
    the complete snapshot.  This is an explicit scope-boundary experiment, not
    a claim about arbitrary relationships outside the supported SQL fragment.
    """

    options = dict(repair_options or {})
    rows: list[dict[str, Any]] = []
    for dataset, dataset_assets in assets.groupby("dataset", sort=False):
        ordered_ids = dataset_assets["asset_id"].astype(str).tolist()
        dataset_relations = relations[relations["dataset"] == dataset]
        complete = build_snapshot(ordered_ids, dataset_relations, epoch=1)
        policy_map = (
            policy[policy["dataset"] == dataset]
            .set_index("asset_id")["target_price"]
            .to_dict()
        )
        center = np.asarray([policy_map[item] for item in ordered_ids], dtype=float)
        if not independent_verify(complete, center, tolerance=tolerance).passed:
            raise ValueError(f"Controlled policy for {dataset} is not feasible")
        relation_types = sorted(
            set(complete.row_metadata["constraint_type"])
            & {
                "decomposition",
                "projection_derivation",
                "filter_derivation",
            }
        )
        for budget in budgets:
            for relation_type in relation_types:
                attack_center, attack_center_policy = (
                    relation_family_attack_center(
                        complete,
                        center,
                        relation_type,
                        tolerance=tolerance,
                    )
                )
                attacks = exact_box_row_attacks(
                    complete,
                    attack_center,
                    budget_fraction=float(budget),
                    limit=int(cases_per_relation_type),
                    tolerance=tolerance,
                    include_types=[relation_type],
                )
                if len(attacks) < int(cases_per_relation_type):
                    raise RuntimeError(
                        f"Only {len(attacks)} unique omitted-row cases are "
                        f"attackable for {dataset}/{budget}/{relation_type}; "
                        f"required {cases_per_relation_type}"
                    )
                for case_rank, attack in enumerate(attacks):
                    row_index = int(attack["row_index"])
                    row_meta = complete.row_metadata.iloc[row_index]
                    incomplete = snapshot_without_rows(
                        complete,
                        [row_index],
                    )
                    candidate = np.asarray(attack["candidate"], dtype=float)
                    complete_before = independent_verify(
                        complete,
                        candidate,
                        tolerance=tolerance,
                    )
                    incomplete_before = independent_verify(
                        incomplete,
                        candidate,
                        tolerance=tolerance,
                    )
                    enforcement = run_enforcement_method(
                        incomplete,
                        candidate,
                        method="arbiter_active",
                        tolerance=tolerance,
                        **options,
                    )
                    complete_after = independent_verify(
                        complete,
                        enforcement.prices,
                        tolerance=tolerance,
                    )
                    start = complete.matrix.indptr[row_index]
                    end = complete.matrix.indptr[row_index + 1]
                    columns = complete.matrix.indices[start:end]
                    coefficients = complete.matrix.data[start:end]
                    omitted_residual_after = float(
                        np.dot(coefficients, enforcement.prices[columns])
                        - complete.bounds[row_index]
                    )
                    released = not enforcement.rejected
                    gap_survives = bool(
                        released
                        and enforcement.verification.passed
                        and omitted_residual_after > tolerance
                    )
                    rows.append(
                        {
                            "dataset": dataset,
                            "budget_fraction": float(budget),
                            "case_rank": int(case_rank),
                            "case_id": hashlib.sha256(
                                (
                                    f"{dataset}|{budget}|{relation_type}|"
                                    f"{attack['row_id']}"
                                ).encode("utf-8")
                            ).hexdigest()[:24],
                            "relation_id": str(row_meta["row_id"]),
                            "relation_type": relation_type,
                            "attack_center_policy": attack_center_policy,
                            "attack_center_digest": hashlib.sha256(
                                np.asarray(
                                    attack_center,
                                    dtype="<f8",
                                ).tobytes(order="C")
                            ).hexdigest(),
                            "target_id": str(row_meta["target_id"]),
                            "source_id": str(row_meta["source_id"]),
                            "component_ids": str(row_meta["component_ids"]),
                            "complete_constraint_digest": (
                                complete.constraint_digest
                            ),
                            "incomplete_constraint_digest": (
                                incomplete.constraint_digest
                            ),
                            "certified_target_residual": float(
                                attack["certified_target_residual"]
                            ),
                            "complete_violations_before": (
                                complete_before.violations
                            ),
                            "incomplete_violations_before": (
                                incomplete_before.violations
                            ),
                            "incomplete_violations_after": (
                                enforcement.verification.violations
                            ),
                            "complete_violations_after": (
                                complete_after.violations
                            ),
                            "omitted_residual_after": omitted_residual_after,
                            "released_by_incomplete_gate": released,
                            "fallback_used": enforcement.fallback_used,
                            "gate_ms": enforcement.runtime_ms,
                            "l1_adjustment": enforcement.l1_adjustment,
                            "gap_survives": gap_survives,
                        }
                    )
    return pd.DataFrame(rows)


def run_scale_benchmark(
    *,
    assets: pd.DataFrame,
    relations: pd.DataFrame,
    policy: pd.DataFrame,
    scales: Sequence[int],
    repetitions: int,
    warmups: int,
    attack_cases_per_relation_type: int,
    methods: Sequence[str] = (
        "reject_only",
        "arbiter_active",
        "full_cyclic",
        "osqp_qp",
    ),
    attack_budget: float = 0.25,
    tolerance: float = DEFAULT_TOLERANCE,
    repair_options: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    options = dict(repair_options or {})
    trial_rows = []
    build_rows = []

    for dataset, dataset_assets in assets.groupby("dataset", sort=False):
        dataset_assets = dataset_assets.reset_index(drop=True)
        dataset_relations = relations[relations["dataset"] == dataset]
        dataset_policy = (
            policy[policy["dataset"] == dataset]
            .set_index("asset_id")["target_price"]
            .to_dict()
        )
        for scale in scales:
            if scale > len(dataset_assets):
                raise ValueError(
                    f"requested scale {scale} exceeds {dataset} asset count"
                )
            ordered_ids = balanced_relation_closed_asset_ids(
                dataset_assets,
                dataset_relations,
                int(scale),
            )
            subset_assets = (
                dataset_assets.set_index("asset_id").loc[ordered_ids].reset_index()
            )
            subset_relations = relation_subset(
                dataset_relations,
                ordered_ids,
            )
            if subset_relations.empty:
                raise RuntimeError(f"No closed relations for {dataset}/{scale}")
            expected_types = set(dataset_relations["relation_type"].astype(str))
            observed_types = set(subset_relations["relation_type"].astype(str))
            if observed_types != expected_types:
                raise RuntimeError(
                    f"Scale subset lost relation families for {dataset}/{scale}: "
                    f"expected {sorted(expected_types)}, observed {sorted(observed_types)}"
                )

            build_times = []
            snapshot = None
            for build_rep in range(repetitions):
                started = time.perf_counter_ns()
                built = build_snapshot(ordered_ids, subset_relations, epoch=1)
                snapshot_build_ms = (time.perf_counter_ns() - started) / 1e6
                build_times.append(snapshot_build_ms)
                snapshot = built
                sparse_numeric_bytes = int(
                    built.matrix.data.nbytes
                    + built.matrix.indices.nbytes
                    + built.matrix.indptr.nbytes
                    + built.bounds.nbytes
                )
                metadata_bytes = int(
                    built.row_metadata.memory_usage(index=True, deep=True).sum()
                )
                build_rows.append(
                    {
                        "dataset": dataset,
                        "scale": int(scale),
                        "repetition": int(build_rep),
                        "snapshot_build_ms": float(snapshot_build_ms),
                        "assets": built.assets,
                        "constraints": built.constraints,
                        "nonzeros": built.nonzeros,
                        "density": built.density,
                        "sparse_numeric_bytes": sparse_numeric_bytes,
                        "row_metadata_bytes": metadata_bytes,
                        "snapshot_estimated_bytes": (
                            sparse_numeric_bytes + metadata_bytes
                        ),
                        "constraint_digest": built.constraint_digest,
                    }
                )
            assert snapshot is not None
            center = np.asarray(
                [dataset_policy[asset_id] for asset_id in ordered_ids], dtype=float
            )
            if not independent_verify(snapshot, center, tolerance=tolerance).passed:
                raise ValueError(f"Subset policy is infeasible for {dataset}/{scale}")
            relation_types = sorted(expected_types)
            per_type_centers = {
                relation_type: relation_family_attack_center(
                    snapshot,
                    center,
                    relation_type,
                    tolerance=tolerance,
                )
                for relation_type in relation_types
            }
            per_type_attacks = {
                relation_type: exact_box_row_attacks(
                    snapshot,
                    per_type_centers[relation_type][0],
                    budget_fraction=attack_budget,
                    limit=int(attack_cases_per_relation_type),
                    tolerance=tolerance,
                    include_types=[relation_type],
                )
                for relation_type in relation_types
            }
            insufficient = {
                relation_type: len(pool)
                for relation_type, pool in per_type_attacks.items()
                if len(pool) < int(attack_cases_per_relation_type)
            }
            if insufficient:
                raise RuntimeError(
                    f"Insufficient unique scale attacks for {dataset}/{scale}: "
                    f"{insufficient}"
                )
            attack_pool = [
                attack
                for relation_type in relation_types
                for attack in per_type_attacks[relation_type]
            ]

            method_list = list(methods)
            for case_index, attack in enumerate(attack_pool):
                attack_center_policy = per_type_centers[
                    attack["constraint_type"]
                ][1]
                candidate = attack["candidate"]
                candidate_digest = hashlib.sha256(
                    np.asarray(candidate, dtype="<f8").tobytes(order="C")
                ).hexdigest()
                attack_case_id = hashlib.sha256(
                    (
                        f"{dataset}|{scale}|{attack_budget}|"
                        f"{attack['constraint_type']}|{attack['row_id']}"
                    ).encode("utf-8")
                ).hexdigest()[:24]
                precheck = independent_verify(
                    snapshot, candidate, tolerance=tolerance
                )
                # Counterbalance warm-up and trial order so one method is not
                # always favored by cache, allocator, or thermal state. Each
                # repetition remains paired on the same snapshot and candidate.
                warmup_order = method_list[
                    case_index % len(method_list) :
                ] + method_list[: case_index % len(method_list)]
                for _ in range(warmups):
                    for method in warmup_order:
                        run_enforcement_method(
                            snapshot,
                            candidate,
                            method=method,
                            tolerance=tolerance,
                            **options,
                        )
                for repetition in range(repetitions):
                    rotation = (case_index + repetition) % len(method_list)
                    trial_order = (
                        method_list[rotation:] + method_list[:rotation]
                    )
                    for order_position, method in enumerate(trial_order):
                        result = run_enforcement_method(
                            snapshot,
                            candidate,
                            method=method,
                            tolerance=tolerance,
                            **options,
                        )
                        snapshot_build_ms = build_times[
                            repetition % len(build_times)
                        ]
                        trial_rows.append(
                            {
                                "dataset": dataset,
                                "scale": int(scale),
                                "attack_case_id": attack_case_id,
                                "case_index": int(case_index),
                                "repetition": int(repetition),
                                "execution_order": int(order_position),
                                "method": method,
                                "assets": snapshot.assets,
                                "constraints": snapshot.constraints,
                                "nonzeros": snapshot.nonzeros,
                                "density": snapshot.density,
                                "target_row_id": attack["row_id"],
                                "target_constraint_type": attack[
                                    "constraint_type"
                                ],
                                "attack_center_policy": attack_center_policy,
                                "candidate_digest": candidate_digest,
                                "certified_target_residual": attack[
                                    "certified_target_residual"
                                ],
                                "violations_before": precheck.violations,
                                "max_residual_before": precheck.max_residual,
                                "snapshot_build_ms": float(snapshot_build_ms),
                                "method_runtime_ms": result.runtime_ms,
                                "cached_total_ms": result.runtime_ms,
                                "snapshot_build_plus_gate_ms": (
                                    snapshot_build_ms + result.runtime_ms
                                ),
                                "verified": result.verification.passed,
                                "violations_after": result.verification.violations,
                                "max_residual_after": (
                                    result.verification.max_residual
                                ),
                                "fallback_used": result.fallback_used,
                                "rejected": result.rejected,
                                "unsafe_release": bool(
                                    not result.rejected
                                    and not result.verification.passed
                                ),
                                "l1_adjustment": result.l1_adjustment,
                                "sweeps": result.sweeps,
                                "projected_rows": result.projected_rows,
                                "solver": result.solver,
                                "constraint_digest": snapshot.constraint_digest,
                            }
                        )
    return pd.DataFrame(trial_rows), pd.DataFrame(build_rows)


def summarize_scale_trials(trials: pd.DataFrame) -> pd.DataFrame:
    if trials.empty:
        return pd.DataFrame()
    return (
        trials.groupby(["dataset", "scale", "method"], as_index=False)
        .agg(
            trials=("method_runtime_ms", "size"),
            attack_cases=("attack_case_id", "nunique"),
            timing_repetitions=("repetition", "nunique"),
            runtime_p50_ms=("method_runtime_ms", "median"),
            runtime_p95_ms=(
                "method_runtime_ms",
                lambda values: float(np.quantile(values, 0.95)),
            ),
            runtime_p99_ms=(
                "method_runtime_ms",
                lambda values: float(np.quantile(values, 0.99)),
            ),
            snapshot_build_plus_gate_p50_ms=(
                "snapshot_build_plus_gate_ms",
                "median",
            ),
            snapshot_build_plus_gate_p95_ms=(
                "snapshot_build_plus_gate_ms",
                lambda values: float(np.quantile(values, 0.95)),
            ),
            verified_rate=("verified", "mean"),
            fallback_rate=("fallback_used", "mean"),
            rejection_rate=("rejected", "mean"),
            unsafe_release_rate=("unsafe_release", "mean"),
            mean_violations_before=("violations_before", "mean"),
            mean_violations_after=("violations_after", "mean"),
            max_residual_after=("max_residual_after", "max"),
            mean_l1_adjustment=("l1_adjustment", "mean"),
        )
        .sort_values(["dataset", "scale", "method"])
        .reset_index(drop=True)
    )


def run_incremental_benchmark(
    *,
    asset_ids: Sequence[str],
    relations: pd.DataFrame,
    repetitions: int,
    batch_sizes: Sequence[int] = (1, 8, 32),
) -> pd.DataFrame:
    if len(relations) < 2:
        raise ValueError("Incremental benchmark needs at least two relations")
    ordered = relations.sort_values("relation_id").reset_index(drop=True)

    def row_multiset(snapshot: ConstraintSnapshot) -> list[tuple[Any, ...]]:
        signatures = []
        for row in range(snapshot.constraints):
            start, end = snapshot.matrix.indptr[row], snapshot.matrix.indptr[row + 1]
            entries = tuple(
                (int(column), float(value))
                for column, value in zip(
                    snapshot.matrix.indices[start:end],
                    snapshot.matrix.data[start:end],
                    strict=True,
                )
            )
            metadata = tuple(
                sorted(
                    (
                        str(key),
                        "" if pd.isna(value) else str(value),
                    )
                    for key, value in snapshot.row_metadata.iloc[row].items()
                )
            )
            signatures.append((entries, float(snapshot.bounds[row]), metadata))
        return sorted(signatures, key=str)

    rows = []
    usable_batch_sizes = sorted(
        {
            int(size)
            for size in batch_sizes
            if int(size) >= 1 and int(size) < len(ordered)
        }
    )
    if not usable_batch_sizes:
        raise ValueError("No incremental batch size fits the relation set")
    for batch_size in usable_batch_sizes:
        base_relations = ordered.iloc[:-batch_size]
        addition = ordered.iloc[-batch_size:]
        base_snapshot = build_snapshot(asset_ids, base_relations, epoch=1)
        for repetition in range(repetitions):
            started = time.perf_counter_ns()
            incremental = extend_snapshot(base_snapshot, addition, new_epoch=2)
            incremental_ms = (time.perf_counter_ns() - started) / 1e6
            started = time.perf_counter_ns()
            rebuilt = build_snapshot(asset_ids, ordered, epoch=2)
            rebuild_ms = (time.perf_counter_ns() - started) / 1e6
            same_rows = row_multiset(incremental) == row_multiset(rebuilt)
            rows.append(
                {
                    "batch_relations": batch_size,
                    "repetition": repetition,
                    "assets": len(asset_ids),
                    "base_constraints": base_snapshot.constraints,
                    "updated_constraints": incremental.constraints,
                    "incremental_ms": incremental_ms,
                    "full_rebuild_ms": rebuild_ms,
                    "speedup": rebuild_ms / max(incremental_ms, 1e-12),
                    "same_shape": incremental.matrix.shape == rebuilt.matrix.shape,
                    "same_constraint_count": (
                        incremental.constraints == rebuilt.constraints
                    ),
                    "same_constraint_row_multiset": same_rows,
                }
            )
    return pd.DataFrame(rows)


def _fresh_store(
    snapshot: ConstraintSnapshot,
    secret: bytes = b"arbiter-test-secret-32-bytes!!!",
    tolerance: float = DEFAULT_TOLERANCE,
) -> SnapshotStore:
    return SnapshotStore(
        snapshot,
        secret=secret[:32].ljust(32, b"!"),
        verifier_tolerance=tolerance,
    )


def run_fault_injection_campaign(
    *,
    snapshot: ConstraintSnapshot,
    feasible_prices: np.ndarray,
    violating_prices: np.ndarray,
    tolerance: float = DEFAULT_TOLERANCE,
) -> pd.DataFrame:
    """Exercise fail-closed behavior. A passing row means behavior matched intent."""

    rows = []

    def record(
        case: str,
        expected_release: bool,
        actual_release: bool,
        reason: str,
    ) -> None:
        rows.append(
            {
                "case": case,
                "expected_release": expected_release,
                "actual_release": actual_release,
                "reason": reason,
                "passed": bool(expected_release == actual_release),
            }
        )

    store = _fresh_store(snapshot, tolerance=tolerance)
    token = store.issue_token(snapshot, feasible_prices, tolerance=tolerance)
    buffer = AnswerBuffer("answer")
    commit = store.compare_and_commit(token, feasible_prices, buffer)
    record("normal_release", True, commit.released, commit.reason)

    store = _fresh_store(snapshot, tolerance=tolerance)
    token = store.issue_token(snapshot, feasible_prices, tolerance=tolerance)
    replay_buffer = AnswerBuffer("answer")
    first = store.compare_and_commit(token, feasible_prices, replay_buffer)
    second = store.compare_and_commit(token, feasible_prices, AnswerBuffer("answer"))
    record("token_replay_first", True, first.released, first.reason)
    record("token_replay_second", False, second.released, second.reason)

    store = _fresh_store(snapshot, tolerance=tolerance)
    token = store.issue_token(snapshot, feasible_prices, tolerance=tolerance)
    stale_snapshot = ConstraintSnapshot(
        asset_ids=snapshot.asset_ids,
        matrix=snapshot.matrix,
        bounds=snapshot.bounds,
        row_metadata=snapshot.row_metadata,
        epoch=snapshot.epoch + 1,
        constraint_digest=snapshot.constraint_digest,
        asset_digest=snapshot.asset_digest,
    )
    store.advance(stale_snapshot)
    commit = store.compare_and_commit(
        token, feasible_prices, AnswerBuffer("answer")
    )
    record("stale_epoch", False, commit.released, commit.reason)

    tamper_cases = {
        "constraint_digest_tamper": replace_token(
            token, constraint_digest="0" * 64
        ),
        "asset_digest_tamper": replace_token(token, asset_digest="1" * 64),
        "price_digest_tamper": replace_token(token, price_digest="2" * 64),
        "token_mac_tamper": replace_token(token, mac="3" * 64),
    }
    for case, tampered in tamper_cases.items():
        store = _fresh_store(snapshot, tolerance=tolerance)
        commit = store.compare_and_commit(
            tampered, feasible_prices, AnswerBuffer("answer")
        )
        record(case, False, commit.released, commit.reason)

    store = _fresh_store(snapshot, tolerance=tolerance)
    token = store.issue_token(snapshot, feasible_prices, tolerance=tolerance)
    changed = feasible_prices.copy()
    changed[0] += 1.0
    commit = store.compare_and_commit(token, changed, AnswerBuffer("answer"))
    record("price_vector_tamper", False, commit.released, commit.reason)

    empty_relations = pd.DataFrame(
        columns=[
            "relation_id",
            "relation_type",
            "target_id",
            "component_ids",
            "derived_id",
            "source_id",
            "sound",
        ]
    )
    underconstrained = build_snapshot(
        snapshot.asset_ids,
        empty_relations,
        epoch=snapshot.epoch,
    )
    under_store = _fresh_store(underconstrained, tolerance=tolerance)
    under_token = under_store.issue_token(
        underconstrained,
        feasible_prices,
        tolerance=tolerance,
    )
    full_store = _fresh_store(snapshot, tolerance=tolerance)
    commit = full_store.compare_and_commit(
        under_token,
        feasible_prices,
        AnswerBuffer("answer"),
    )
    record("missing_constraint_rows", False, commit.released, commit.reason)

    store = _fresh_store(snapshot, tolerance=tolerance)
    token = store.issue_token(snapshot, feasible_prices, tolerance=tolerance)
    store.restart_with_new_secret()
    commit = store.compare_and_commit(
        token, feasible_prices, AnswerBuffer("answer")
    )
    record("process_restart_secret_rotation", False, commit.released, commit.reason)

    for case, bad_prices in [
        ("nan_input", np.full(snapshot.assets, np.nan)),
        ("infinity_input", np.full(snapshot.assets, np.inf)),
        ("missing_price", feasible_prices[:-1]),
    ]:
        verification = independent_verify(snapshot, bad_prices, tolerance=tolerance)
        record(case, False, verification.passed, verification.reason)

    reordered_ids = list(reversed(snapshot.asset_ids))
    reordered_prices = np.asarray(feasible_prices, dtype=float)[::-1]
    reordered = independent_verify(
        snapshot,
        reordered_prices,
        ordered_asset_ids=reordered_ids,
        tolerance=tolerance,
    )
    record(
        "asset_order_reversal",
        False,
        reordered.passed,
        reordered.reason,
    )

    # Tolerance is a store policy, not a caller-controlled escape hatch.
    for case, invalid_tolerance in [
        ("nan_tolerance", math.nan),
        ("infinite_tolerance", math.inf),
        ("negative_tolerance", -1.0),
        ("mismatched_large_tolerance", 1e9),
    ]:
        store = _fresh_store(snapshot, tolerance=tolerance)
        try:
            store.issue_token(
                snapshot,
                feasible_prices,
                tolerance=invalid_tolerance,
            )
            issued = True
            reason = "invalid_tolerance_accepted"
        except Exception as exc:
            issued = False
            reason = f"{type(exc).__name__}: {exc}"
        record(case, False, issued, reason)

    # A snapshot that is mutated after token issuance must fail the final
    # identity check even if its epoch and token fields are unchanged.
    mutable_matrix = snapshot.matrix.copy()
    mutable_bounds = snapshot.bounds.copy()
    mutable_snapshot = ConstraintSnapshot(
        asset_ids=snapshot.asset_ids,
        matrix=mutable_matrix,
        bounds=mutable_bounds,
        row_metadata=snapshot.row_metadata.copy(deep=True),
        epoch=snapshot.epoch,
        constraint_digest=snapshot.constraint_digest,
        asset_digest=snapshot.asset_digest,
    )
    store = _fresh_store(mutable_snapshot, tolerance=tolerance)
    token = store.issue_token(
        mutable_snapshot,
        feasible_prices,
        tolerance=tolerance,
    )
    mutable_matrix.data[0] += 0.25
    commit = store.compare_and_commit(
        token,
        feasible_prices,
        AnswerBuffer("answer"),
    )
    record("snapshot_mutation_after_verify", False, commit.released, commit.reason)

    failed = run_enforcement_method(
        snapshot,
        violating_prices,
        method="arbiter_active",
        tolerance=tolerance,
        max_sweeps=0,
        allow_fallback=True,
        force_fallback_failure=True,
    )
    record(
        "fallback_failure",
        False,
        not failed.rejected,
        failed.error or failed.verification.reason,
    )

    timed_out = run_enforcement_method(
        snapshot,
        violating_prices,
        method="arbiter_active",
        tolerance=tolerance,
        max_sweeps=0,
        allow_fallback=True,
        osqp_max_iter=1,
    )
    record(
        "solver_iteration_limit",
        False,
        not timed_out.rejected,
        timed_out.error or timed_out.verification.reason,
    )
    return pd.DataFrame(rows)


def run_stale_epoch_concurrency_campaign(
    *,
    snapshot: ConstraintSnapshot,
    feasible_prices: np.ndarray,
    trials: int,
    tolerance: float = DEFAULT_TOLERANCE,
) -> pd.DataFrame:
    """Test ordered-stale and racing update/commit schedules.

    A race is valid only if the lock linearizes it to either a release under the
    old epoch or a stale-epoch rejection under the new epoch. No other result is
    accepted.
    """

    rows = []
    next_snapshot = ConstraintSnapshot(
        asset_ids=snapshot.asset_ids,
        matrix=snapshot.matrix,
        bounds=snapshot.bounds,
        row_metadata=snapshot.row_metadata,
        epoch=snapshot.epoch + 1,
        constraint_digest=snapshot.constraint_digest,
        asset_digest=snapshot.asset_digest,
    )
    for schedule in ["update_before_commit", "simultaneous_race"]:
        for trial in range(trials):
            store = _fresh_store(snapshot, tolerance=tolerance)
            token = store.issue_token(
                snapshot, feasible_prices, tolerance=tolerance
            )
            buffer = AnswerBuffer({"trial": trial, "schedule": schedule})
            result_holder: dict[str, Any] = {}
            start_barrier = threading.Barrier(3)
            update_done = threading.Event()

            def updater() -> None:
                if schedule == "simultaneous_race":
                    start_barrier.wait(timeout=5)
                store.advance(next_snapshot)
                update_done.set()

            def committer() -> None:
                if schedule == "update_before_commit":
                    update_done.wait(timeout=5)
                else:
                    start_barrier.wait(timeout=5)
                result_holder["commit"] = store.compare_and_commit(
                    token, feasible_prices, buffer
                )

            update_thread = threading.Thread(target=updater)
            commit_thread = threading.Thread(target=committer)
            update_thread.start()
            commit_thread.start()
            if schedule == "simultaneous_race":
                start_barrier.wait(timeout=5)
            update_thread.join(timeout=5)
            commit_thread.join(timeout=5)
            commit = result_holder.get("commit")
            if schedule == "update_before_commit":
                passed = bool(
                    commit
                    and not commit.released
                    and commit.reason == "stale_epoch"
                    and not buffer.released
                )
            else:
                passed = bool(
                    commit
                    and (
                        (
                            commit.released
                            and commit.reason == "released"
                            and buffer.released
                        )
                        or (
                            not commit.released
                            and commit.reason == "stale_epoch"
                            and not buffer.released
                        )
                    )
                )
            rows.append(
                {
                    "schedule": schedule,
                    "trial": trial,
                    "epoch_before": snapshot.epoch,
                    "final_epoch": store.read().epoch,
                    "released": bool(commit.released) if commit else False,
                    "reason": commit.reason if commit else "thread_timeout",
                    "passed": passed,
                }
            )
    return pd.DataFrame(rows)
