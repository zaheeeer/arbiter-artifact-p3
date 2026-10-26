"""Compare deterministic columns of the Notebook 07 Colab run with a second execution.

Usage: python3 -I compare_second_execution.py COLAB_OUTPUTS SECOND_OUTPUTS
Timing columns are excluded. OSQP rows are excluded where the offline run used
the OSQP stand-in. Prints, per file and column, how many keyed rows agree.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

C, L = map(Path, sys.argv[1:3])
NON_OSQP = {"reject_only", "arbiter_active", "full_cyclic"}

SPECS = {
    "s01_natural_gate": (["dataset", "model", "seed", "tolerance"], None, ["gate_ms", "fallback_ms"]),
    "s01_natural_top_relations": (["dataset", "model", "seed", "tolerance", "rank"], None, []),
    "s01_heavy_raw": (["dataset", "model", "seed", "repetition", "method"], ("method", NON_OSQP), ["runtime_ms", "execution_order", "proposal_index"]),
    "s02_attacks": (["attack_id"], None, ["repair_ms", "fallback_ms"]),
    "s02_attack_deltas": (["attack_id", "asset_id"], None, []),
    "s02_attack_records": (["attack_id"], None, []),
    "s03_gap_cases": (["dataset", "case_id"], None, ["gate_ms"]),
    "s04_build_raw": (["dataset", "scale", "repetition"], None, ["snapshot_build_ms"]),
    "s04_scale_raw": (["dataset", "scale", "attack_case_id", "repetition", "method"], ("method", NON_OSQP), ["runtime_ms", "snapshot_build_ms", "snapshot_build_plus_gate_ms"]),
    "s05_ablation_raw": (["ablation", "dataset", "attack_case_id", "repetition", "method"], None, ["runtime_ms", "snapshot_build_ms", "snapshot_build_plus_gate_ms"]),
    "s06_dense_catalog": (["tolerance", "max_sweeps", "dataset", "relation_family", "case_id"], None, []),
    "s06_dense_raw": (["tolerance", "dataset", "relation_family", "case_id", "active_row_budget", "max_sweeps", "variant", "repetition"], None, ["runtime_ms", "fallback_ms", "active_path_ms", "execution_order"]),
    "s06_dense_optimized": (["tolerance", "dataset", "relation_family", "case_id", "active_row_budget", "max_sweeps", "violated_only_sweeps"], None, ["runtime_ms", "fallback_ms"]),
    "s07_fallback_conditions": (["dataset", "relation_family", "case_id", "active_row_budget"], None, []),
    "s07_fallback_raw": (["dataset", "relation_family", "case_id", "active_row_budget", "fallback_sweeps", "method", "repetition"], None, ["runtime_ms", "kernel_reported_ms", "execution_order"]),
    "s08_coverage_raw": (["dataset", "encoding", "model", "seed"], None, ["gate_ms"]),
    "s09_shape_policy": (["dataset", "shape"], None, []),
    "s09_shape_model_metrics": (["dataset", "shape", "model", "seed"], None, []),
    "s09_shape_gate": (["dataset", "shape", "model", "seed"], None, ["gate_ms"]),
    "s09_shape_attacks": (["dataset", "shape", "relation_family", "rank"], None, ["gate_ms"]),
    "s10_recompute_raw": (["dataset", "shape", "repetition"], None, ["recompute_ms", "verify_ms"]),
    "s11_history_raw": (["dataset", "history_size", "buyer", "variant", "proposal", "proposal_seed"], None, ["snapshot_build_ms", "gate_ms"]),
    "s11_history_pinned_diagnostic": (["dataset", "history_size", "buyer", "encoding", "proposal"], None, ["runtime_ms"]),
    "s12_protocol_vectors": (["dataset"], None, ["repair_ms_1e7", "repair_ms_1e12"]),
    "s12_v4_faults": (["dataset", "case"], None, []),
    "s12_closure_faults": (["dataset", "case", "phase"], None, []),
    "s12_closure_delivery": (["dataset", "case"], None, []),
    "final_sweep_budget_curve": (["family"], None, []),
}


def load(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip", keep_default_na=False, na_values=[""])


total_rows = total_cells = total_mismatch_cells = 0
for name, (keys, row_filter, skip) in SPECS.items():
    a, b = load(C / f"{name}.csv"), load(L / f"{name}.csv")
    if row_filter is not None:
        col, allowed = row_filter
        a, b = a[a[col].isin(allowed)], b[b[col].isin(allowed)]
    dup_a, dup_b = a.duplicated(keys).sum(), b.duplicated(keys).sum()
    m = a.merge(b, on=keys, how="outer", suffixes=("_c", "_l"), indicator=True)
    both = m[m["_merge"] == "both"]
    cols = [c for c in a.columns if c not in keys and c not in skip and c in b.columns]
    diffs = []
    for c in cols:
        x, y = both[f"{c}_c"], both[f"{c}_l"]
        if pd.api.types.is_float_dtype(x) and pd.api.types.is_float_dtype(y):
            eq = (x == y) | (x.isna() & y.isna())
            scale = np.maximum(np.abs(x.to_numpy()), np.abs(y.to_numpy()))
            with np.errstate(invalid="ignore", divide="ignore"):
                rel = np.where(scale > 0, np.abs(x.to_numpy() - y.to_numpy()) / scale, 0.0)
            rel = np.nan_to_num(rel, nan=0.0)
            if (~eq).sum():
                diffs.append(f"{c}: {int((~eq).sum())} differ (max rel {np.max(rel):.2e})")
        else:
            # Missing on both sides counts as equal; numbers compare as numbers
            # (an integer column read as 9 on one side and 9.0 on the other is equal).
            xn, yn = pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")
            numeric = xn.notna() & yn.notna()
            eq = (x.isna() & y.isna()) | (numeric & (xn == yn)) | (~numeric & (x.astype(str) == y.astype(str)))
            if (~eq).sum():
                ex = both.loc[~eq, [f"{c}_c", f"{c}_l"]].head(2).to_dict("records")
                diffs.append(f"{c}: {int((~eq).sum())} differ e.g. {ex}")
        total_cells += len(both)
        total_mismatch_cells += int((~eq).sum())
    total_rows += len(both)
    status = "IDENTICAL" if not diffs and len(both) == len(a) == len(b) else "DIFFERS"
    print(f"{name:32s} rows colab={len(a)} local={len(b)} matched={len(both)} dupkeys={dup_a}/{dup_b} cols={len(cols)} -> {status}")
    for d in diffs:
        print("    ", d[:400])
print(f"TOTAL matched rows {total_rows}; compared cells {total_cells}; differing cells {total_mismatch_cells}")
