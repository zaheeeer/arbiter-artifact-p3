"""Independent recomputation of the Notebook 07 summaries from raw rows.

Usage: python3 -I audit_recompute.py RUN_DIR AUDITED_V4_DIR

RUN_DIR is the extracted 07_revision_full_colab-cpu-primary folder. The script
does not import any Arbiter module. It re-derives every summary table from the
raw CSVs with its own code, reproduces the bootstrap intervals from their
documented seeds, compares s01/s02/s03 with the audited-v4 records, and
recomputes the claim-audit values. It prints one line per check.
"""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RUN, V4 = map(Path, sys.argv[1:3])
O = RUN / "outputs"
TOL = 1e-12
results = []


def rd(path):
    return pd.read_csv(path, float_precision="round_trip")


def b(series):
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])


def seed_of(*parts):
    digest = hashlib.sha256("\x1f".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2 ** 32)


def boot_ci(values, resamples, seed):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return math.nan, math.nan
    if arr.size == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(int(seed))
    meds = np.median(rng.choice(arr, size=(int(resamples), arr.size), replace=True), axis=1)
    lo, hi = np.quantile(meds, [0.025, 0.975])
    return float(lo), float(hi)


def qtl(values, q, method="linear"):
    arr = np.asarray(values, dtype=float)
    return float(np.quantile(arr, q, method=method)) if arr.size else math.nan


def compare(name, mine, theirs, keys, rel=1e-12):
    """Compare two frames on keys; numeric columns within rel, others exact."""
    mine = mine.copy()
    theirs = theirs.copy()
    for k in keys:
        if pd.api.types.is_float_dtype(theirs[k]) or pd.api.types.is_float_dtype(mine[k]):
            mine[k] = mine[k].astype(float)
            theirs[k] = theirs[k].astype(float)
        else:
            mine[k] = mine[k].astype(str)
            theirs[k] = theirs[k].astype(str)
    cols = [c for c in mine.columns if c not in keys and c in theirs.columns]
    m = mine.merge(theirs, on=keys, how="outer", suffixes=("_mine", "_nb"), indicator=True)
    only = int((m["_merge"] != "both").sum())
    m = m[m["_merge"] == "both"]
    bad, worst = [], 0.0
    for c in cols:
        x, y = m[f"{c}_mine"], m[f"{c}_nb"]
        xn = pd.to_numeric(x, errors="coerce")
        yn = pd.to_numeric(y.astype(float) if y.dtype == bool else y, errors="coerce")
        if x.dtype == bool or y.dtype == bool:
            xn, yn = pd.to_numeric(x.astype(float), errors="coerce"), pd.to_numeric(y.astype(float), errors="coerce")
        numeric = xn.notna().sum() == x.notna().sum() and yn.notna().sum() == y.notna().sum() and x.notna().any()
        if numeric:
            a, c2 = xn.to_numpy(float), yn.to_numpy(float)
            both_nan = np.isnan(a) & np.isnan(c2)
            scale = np.maximum(np.abs(a), np.abs(c2))
            diff = np.where(both_nan, 0.0, np.abs(a - c2) / np.where(scale > 0, scale, 1.0))
            diff = np.nan_to_num(diff, nan=1.0)
            worst = max(worst, float(diff.max()) if diff.size else 0.0)
            n = int((diff > rel).sum())
        else:
            n = int((x.astype(str) != y.astype(str)).sum())
        if n:
            bad.append(f"{c}:{n}")
    ok = not bad and only == 0 and len(m) == len(mine) == len(theirs)
    results.append((name, ok))
    print(f"[{'OK ' if ok else 'BAD'}] {name}: rows {len(m)}/{len(theirs)} unmatched {only} max rel diff {worst:.1e} {' '.join(bad)}")


cfg = json.loads((RUN / "revision_run_config.json").read_text())
R = int(cfg["bootstrap_resamples"])

# ---------- s01 natural ----------
nat = rd(O / "s01_natural_gate.csv")
for c in ["released", "rejected", "fallback_used", "unsafe_release"]:
    nat[c] = b(nat[c])


def nat_agg(g):
    return pd.Series({
        "releases": int(g.released.sum()), "proposals": len(g), "fallback": int(g.fallback_used.sum()),
        "rejected": int(g.rejected.sum()), "unsafe": int(g.unsafe_release.sum()), "sweeps_max": g.sweeps.max(),
        "max_saving_vs_list": g.max_saving_vs_list.max(),
        "releases_with_saving_gt_1pct": int((g.n_saving_gt_1pct > 0).sum()),
        "releases_with_saving_gt_0p1pct": int((g.n_saving_gt_0p1pct > 0).sum()),
        "max_saving_conservative": g.max_saving_conservative.max(),
        "releases_with_conservative_gt_1pct": int((g.n_conservative_gt_1pct > 0).sum()),
        "max_residual_after": g.max_residual_after.max(), "gate_p50_ms": float(np.median(g.gate_ms)),
    })


mine = nat.groupby("tolerance").apply(nat_agg, include_groups=False).reset_index()
compare("s01 tolerance_overall", mine, rd(O / "s01_tolerance_overall.csv"), ["tolerance"])

rows = []
for (d, mo, t), g in nat.groupby(["dataset", "model", "tolerance"]):
    rows.append({"dataset": d, "model": mo, "tolerance": t, "proposals": len(g),
                 "violations_before_min": g.violations_before.min(), "violations_before_max": g.violations_before.max(),
                 "released_rate": g.released.mean(), "fallback_rate": g.fallback_used.mean(), "rejected_rate": g.rejected.mean(),
                 "unsafe_releases": int(g.unsafe_release.sum()), "sweeps_max": g.sweeps.max(),
                 "gate_p50_ms": float(np.median(g.gate_ms)), "gate_max_ms": g.gate_ms.max(),
                 "max_residual_after": g.max_residual_after.max(), "max_saving_vs_list": g.max_saving_vs_list.max(),
                 "releases_with_saving_gt_1pct": int((g.n_saving_gt_1pct > 0).sum()),
                 "releases_with_saving_gt_0p1pct": int((g.n_saving_gt_0p1pct > 0).sum()),
                 "max_saving_conservative": g.max_saving_conservative.max(),
                 "releases_with_conservative_gt_1pct": int((g.n_conservative_gt_1pct > 0).sum())})
compare("s01 natural_summary", pd.DataFrame(rows), rd(O / "s01_natural_summary.csv"), ["dataset", "model", "tolerance"])

# s01 at 1e-7 versus audited records
aud = rd(V4 / "01_proposals_attacks" / "natural_proposal_gate.csv")
n7 = nat[nat.tolerance == 1e-7]
m = aud.merge(n7, on=["dataset", "model", "seed"], suffixes=("_v4", "_r"))
same = all((m[f"{c}_v4"] == m[f"{c}_r"]).all() for c in ["violations_before", "max_residual_before", "violations_after", "max_residual_after", "l1_adjustment"])
same = same and (b(m.released_v4) == m.released_r).all() and (b(m.fallback_used_v4) == m.fallback_used_r).all()
results.append(("s01 1e-7 equals audited natural_proposal_gate", bool(same and len(m) == 45)))
print(f"[{'OK ' if same and len(m) == 45 else 'BAD'}] s01 at 1e-7 equals audited natural_proposal_gate.csv exactly: {len(m)}/45 rows")


# ---------- method-trial summaries ----------
def method_summary(raw, groups):
    raw = raw.copy()
    for c in ["verified", "rejected", "unsafe_release", "fallback_used"]:
        raw[c] = b(raw[c])
    rows = []
    for keys, g in raw.groupby(groups):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rt = g.runtime_ms.to_numpy(float)
        rows.append({**dict(zip(groups, keys)), "trials": len(g), "runtime_p50_ms": float(np.median(rt)),
                     "runtime_p95_ms": qtl(rt, 0.95), "runtime_p99_ms": qtl(rt, 0.99), "runtime_mean_ms": float(np.mean(rt)),
                     "verified_rate": g.verified.mean(), "rejected_rate": g.rejected.mean(), "fallback_rate": g.fallback_used.mean(),
                     "unsafe_releases": int(g.unsafe_release.sum()), "sweeps_median": float(np.median(g.sweeps)),
                     "sweeps_max": g.sweeps.max(), "max_residual_after": g.max_residual_after.max(),
                     "mean_l1_adjustment": g.l1_adjustment.mean()})
    return pd.DataFrame(rows)


def paired(raw, cells, unit, ref, baselines):
    raw = raw.copy()
    raw["verified"] = b(raw["verified"])
    rows = []
    for ckeys, cell in raw.groupby(cells):
        ckeys = ckeys if isinstance(ckeys, tuple) else (ckeys,)
        ckeys = tuple(int(k) if isinstance(k, (np.integer,)) else k for k in ckeys)
        refrows = cell[cell.method == ref].set_index([unit, "repetition"])
        for base in baselines:
            br = cell[cell.method == base].set_index([unit, "repetition"])
            j = br[["runtime_ms", "verified"]].join(refrows[["runtime_ms"]], rsuffix="_ref", how="inner")
            j["ratio"] = j.runtime_ms / j.runtime_ms_ref
            for label, sub in [("all_trials", j), ("baseline_verified_trials", j[j.verified])]:
                um = sub.groupby(level=0)["ratio"].median().sort_index().to_numpy(float)
                med = float(np.median(um)) if um.size else math.nan
                lo, hi = boot_ci(um, R, seed_of("paired", *ckeys, base, label)) if um.size else (math.nan, math.nan)
                rows.append({**dict(zip(cells, ckeys)), "baseline": base, "subset": label, "paired_trials": len(sub),
                             "units": int(um.size), "baseline_verified_rate": float(j.verified.mean()) if len(j) else math.nan,
                             "speedup_median": med, "speedup_ci95_low": lo, "speedup_ci95_high": hi})
    return pd.DataFrame(rows)


BASE = ["full_cyclic", "osqp_qp", "osqp_qp_v4_stop", "reject_only"]
heavy = rd(O / "s01_heavy_raw.csv")
compare("s01 heavy_summary", method_summary(heavy, ["dataset", "method"]), rd(O / "s01_heavy_summary.csv"), ["dataset", "method"])
compare("s01 heavy_paired (bootstrap reproduced)", paired(heavy, ["dataset"], "proposal_index", "arbiter_active", BASE),
        rd(O / "s01_heavy_paired.csv").drop(columns=["direction"]), ["dataset", "baseline", "subset"])

scale = rd(O / "s04_scale_raw.csv")
compare("s04 scale_summary", method_summary(scale, ["dataset", "scale", "method"]), rd(O / "s04_scale_summary.csv"), ["dataset", "scale", "method"])
compare("s04 scale_paired (bootstrap reproduced)", paired(scale, ["dataset", "scale"], "attack_case_id", "arbiter_active", BASE),
        rd(O / "s04_scale_paired.csv").drop(columns=["direction"]), ["dataset", "scale", "baseline", "subset"])
build = rd(O / "s04_build_raw.csv")
rows = [{"dataset": d, "scale": s, "builds": len(g), "build_p50_ms": float(np.median(g.snapshot_build_ms)),
         "build_p95_ms": qtl(g.snapshot_build_ms, 0.95), "constraints": g.constraints.iloc[0], "nonzeros": g.nonzeros.iloc[0],
         "density": g.density.iloc[0]} for (d, s), g in build.groupby(["dataset", "scale"])]
compare("s04 build_summary", pd.DataFrame(rows), rd(O / "s04_build_summary.csv"), ["dataset", "scale"])
abl = rd(O / "s05_ablation_raw.csv")
compare("s05 ablation_summary", method_summary(abl, ["dataset", "scale", "ablation"]), rd(O / "s05_ablation_summary.csv"), ["dataset", "scale", "ablation"])

# ---------- s02 attacks ----------
att = rd(O / "s02_attacks.csv")
for c in ["released", "fallback_used", "unsafe_release"]:
    att[c] = b(att[c])


def att_agg(groups):
    rows = []
    for keys, g in att.groupby(groups):
        rows.append({**dict(zip(groups, keys)), "attacks": len(g), "distinct_target_rows": g.target_row_id.nunique(),
                     "attack_success_rate": float((g.violations_before > 0).mean()), "violations_before_max": g.violations_before.max(),
                     "released_rate": g.released.mean(), "fallback_rate": g.fallback_used.mean(), "unsafe_release_rate": g.unsafe_release.mean(),
                     "gate_p50_ms": float(np.median(g.repair_ms)), "gate_p95_ms": qtl(g.repair_ms, 0.95),
                     "l1_median": float(np.median(g.l1_adjustment)), "max_post_residual": g.max_residual_after.max(), "sweeps_max": g.sweeps.max()})
    return pd.DataFrame(rows)


compare("s02 attack_summary", att_agg(["dataset", "budget_fraction"]), rd(O / "s02_attack_summary.csv"), ["dataset", "budget_fraction"])
compare("s02 attack_family_summary", att_agg(["dataset", "budget_fraction", "target_constraint_type"]), rd(O / "s02_attack_family_summary.csv"),
        ["dataset", "budget_fraction", "target_constraint_type"])
old = rd(V4 / "01_proposals_attacks" / "certified_row_attacks.csv")
keys = ["dataset", "budget_fraction", "target_constraint_type", "target_row_id"]
mm = att.merge(old, on=keys, suffixes=("_r", "_v4"))
rows = []
for cell, g in mm.groupby(["dataset", "budget_fraction", "target_constraint_type"]):
    rows.append({"dataset": cell[0], "budget_fraction": cell[1], "target_constraint_type": cell[2],
                 "same_target_rows": len(g), "same_release_decision": int((g.released_r == b(g.released_v4)).sum()),
                 "same_violations_before": int((g.violations_before_r == g.violations_before_v4).sum()),
                 "same_sweeps": int((g.sweeps_r == g.sweeps_v4).sum())})
compare("s02 v4 comparison (recomputed from audited CSV)", pd.DataFrame(rows),
        rd(O / "s02_attack_v4_comparison.csv").drop(columns=["cases_1e12", "cases_v4"]), ["dataset", "budget_fraction", "target_constraint_type"])

# ---------- s03 gap ----------
gap = rd(O / "s03_gap_cases.csv")
for c in ["released_by_incomplete_gate", "gap_survives", "fallback_used"]:
    gap[c] = b(gap[c])
rows = []
for (d, bf, rt), g in gap.groupby(["dataset", "budget_fraction", "relation_type"]):
    rows.append({"dataset": d, "budget_fraction": bf, "relation_type": rt, "cases": len(g), "distinct_relations": g.relation_id.nunique(),
                 "candidate_target_violation_rate": float((g.certified_target_residual > TOL).mean()),
                 "incomplete_release_rate": g.released_by_incomplete_gate.mean(), "gap_survival_rate": g.gap_survives.mean(),
                 "gaps_survived": int(g.gap_survives.sum()), "fallback_rate": g.fallback_used.mean(), "gate_p50_ms": float(np.median(g.gate_ms))})
compare("s03 gap_summary", pd.DataFrame(rows), rd(O / "s03_gap_summary.csv"), ["dataset", "budget_fraction", "relation_type"])
ogap = rd(V4 / "01_proposals_attacks" / "omitted_relation_gap_cases.csv")
mg = gap.merge(ogap, on="case_id", suffixes=("_r", "_v4"))
mine = pd.DataFrame([{"cases_1e12": len(gap), "cases_v4": len(ogap), "same_case_ids": len(mg),
                      "same_gap_outcome": int((mg.gap_survives_r == b(mg.gap_survives_v4)).sum()),
                      "gaps_survived_1e12": int(gap.gap_survives.sum()), "gaps_survived_v4": int(b(ogap.gap_survives).sum())}])
mine["k"] = 1
th = rd(O / "s03_gap_v4_comparison.csv"); th["k"] = 1
compare("s03 v4 comparison (recomputed from audited CSV)", mine, th, ["k"])
print("    s03 gap survival by relation type (1e-12 vs v4):",
      gap.groupby("relation_type").gap_survives.sum().to_dict(), b(ogap.gap_survives).groupby(ogap.relation_type).sum().to_dict())

# ---------- s06 dense ----------
dense = rd(O / "s06_dense_raw.csv")
for c in ["budget_binding", "fallback_used", "rejected", "oracle_verified", "unsafe_release"]:
    dense[c] = b(dense[c])
rows = []
for (t, ms, v, bud), g in dense.groupby(["tolerance", "max_sweeps", "variant", "active_row_budget"]):
    rows.append({"tolerance": t, "max_sweeps": ms, "variant": v, "active_row_budget": bud, "trials": len(g), "cases": g.case_id.nunique(),
                 "binding_rate": g.budget_binding.mean(), "fallback_rate": g.fallback_used.mean(), "rejection_rate": g.rejected.mean(),
                 "verified_rate": g.oracle_verified.mean(), "unsafe_releases": int(g.unsafe_release.sum()), "runtime_p50_ms": float(np.median(g.runtime_ms))})
compare("s06 dense_budget_overview", pd.DataFrame(rows), rd(O / "s06_dense_budget_overview.csv"), ["tolerance", "max_sweeps", "variant", "active_row_budget"])
percase = dense.groupby(["tolerance", "max_sweeps", "variant", "active_row_budget", "case_id"])[["fallback_used", "rejected"]].nunique()
print("    s06 outcome constant across repetitions of each case:", bool((percase <= 1).all().all()))
opt = rd(O / "s06_dense_optimized.csv")
for c in ["budget_binding", "fallback_used", "rejected", "oracle_verified", "unsafe_release"]:
    opt[c] = b(opt[c])
rows = []
for (t, ms, vs, bud), g in opt.groupby(["tolerance", "max_sweeps", "violated_only_sweeps", "active_row_budget"]):
    rows.append({"tolerance": t, "max_sweeps": ms, "violated_only_sweeps": vs, "active_row_budget": bud, "cases": g.case_id.nunique(),
                 "fallback_rate": g.fallback_used.mean(), "rejection_rate": g.rejected.mean(), "verified_rate": g.oracle_verified.mean(),
                 "unsafe_releases": int(g.unsafe_release.sum()), "runtime_p50_ms": float(np.median(g.runtime_ms))})
compare("s06 dense_optimized_overview", pd.DataFrame(rows), rd(O / "s06_dense_optimized_overview.csv"), ["tolerance", "max_sweeps", "violated_only_sweeps", "active_row_budget"])

# ---------- s07 fallback kernels ----------
fb = rd(O / "s07_fallback_raw.csv")
fb["verified"] = b(fb["verified"])
K = ["dataset", "relation_family", "case_id", "active_row_budget", "fallback_sweeps"]
rows = []
for keys, g in fb.groupby(K + ["method"]):
    keys = tuple(int(k) if isinstance(k, np.integer) else k for k in keys)
    v = g.runtime_ms.to_numpy(float)
    lo, hi = boot_ci(v, R, seed_of("fallback", *keys))
    rows.append({**dict(zip(K + ["method"], keys)), "repetitions": len(g), "runtime_p50_ms": float(np.median(v)),
                 "runtime_p95_observed_ms": qtl(v, 0.95, "higher"), "runtime_median_ci95_low_ms": lo, "runtime_median_ci95_high_ms": hi,
                 "verified_rate": g.verified.mean(), "sweeps_median": float(np.median(g.sweeps)), "projected_rows_median": float(np.median(g.projected_rows))})
compare("s07 fallback_summary (bootstrap reproduced)", pd.DataFrame(rows), rd(O / "s07_fallback_summary.csv"), K + ["method"])
leg, optk = "legacy_full_scan_cyclic", "optimized_violated_only_cyclic"
rows = []
for keys, g in fb.groupby(K):
    keys = tuple(int(k) if isinstance(k, np.integer) else k for k in keys)
    a = g[g.method == leg].set_index("repetition"); o = g[g.method == optk].set_index("repetition")
    j = a[["runtime_ms", "verified"]].join(o[["runtime_ms", "verified"]], lsuffix="_l", rsuffix="_o")
    ok = j.verified_l & j.verified_o
    valid = (j.runtime_ms_l / j.runtime_ms_o)[ok].to_numpy(float)
    lo, hi = boot_ci(valid, R, seed_of("fallback-speedup", *keys))
    rows.append({**dict(zip(K, keys)), "paired_repetitions": len(j), "both_verified_repetitions": int(valid.size),
                 "speedup_legacy_over_optimized_p50": float(np.median(valid)) if valid.size else math.nan,
                 "speedup_ci95_low": lo, "speedup_ci95_high": hi})
compare("s07 fallback_paired (bootstrap reproduced)", pd.DataFrame(rows), rd(O / "s07_fallback_paired.csv"), K)

# ---------- s08 coverage ----------
cov = rd(O / "s08_coverage_raw.csv")
for c in ["released", "fallback_used", "unsafe_release"]:
    cov[c] = b(cov[c])
rows = []
for (d, e, k), g in cov.groupby(["dataset", "encoding", "encoding_kind"]):
    rows.append({"dataset": d, "encoding": e, "encoding_kind": k, "encoded_share": g.encoded_share.iloc[0], "proposals": len(g),
                 "released_rate": g.released.mean(), "fallback_rate": g.fallback_used.mean(), "unsafe_releases": int(g.unsafe_release.sum()),
                 "releases_with_complete_violations": int((g.complete_violations_after > 0).sum()),
                 "complete_violations_after_median": float(np.median(g.complete_violations_after)),
                 "complete_violations_after_max": g.complete_violations_after.max(), "max_saving_vs_list": g.max_saving_vs_list.max(),
                 "releases_with_saving_gt_1pct": int((g.n_saving_gt_1pct > 0).sum()), "max_saving_conservative": g.max_saving_conservative.max(),
                 "releases_with_conservative_gt_1pct": int((g.n_conservative_gt_1pct > 0).sum()),
                 "exposed_relations_median": float(np.median(g.relations_above_1e12)), "gate_p50_ms": float(np.median(g.gate_ms))})
compare("s08 coverage_summary", pd.DataFrame(rows), rd(O / "s08_coverage_summary.csv"), ["dataset", "encoding", "encoding_kind"])

# ---------- s09 shapes ----------
sg = rd(O / "s09_shape_gate.csv")
for c in ["released", "fallback_used", "unsafe_release"]:
    sg[c] = b(sg[c])
rows = [{"dataset": d, "shape": s, "proposals": len(g), "violations_before_median": float(np.median(g.violations_before)),
         "violations_before_max": g.violations_before.max(), "released_rate": g.released.mean(), "fallback_rate": g.fallback_used.mean(),
         "unsafe_releases": int(g.unsafe_release.sum()), "sweeps_max": g.sweeps.max(), "gate_p50_ms": float(np.median(g.gate_ms)),
         "max_saving_vs_list": g.max_saving_vs_list.max()} for (d, s), g in sg.groupby(["dataset", "shape"])]
compare("s09 shape_gate_summary", pd.DataFrame(rows), rd(O / "s09_shape_gate_summary.csv"), ["dataset", "shape"])
sa = rd(O / "s09_shape_attacks.csv")
for c in ["released", "unsafe_release"]:
    sa[c] = b(sa[c])
rows = [{"dataset": d, "shape": s, "attacks": len(g), "attack_success_rate": float((g.violations_before > 0).mean()),
         "violations_before_max": g.violations_before.max(), "released_rate": g.released.mean(), "unsafe_releases": int(g.unsafe_release.sum())}
        for (d, s), g in sa.groupby(["dataset", "shape"])]
compare("s09 shape_attack_summary", pd.DataFrame(rows), rd(O / "s09_shape_attack_summary.csv"), ["dataset", "shape"])

# ---------- s10 recompute ----------
rc = rd(O / "s10_recompute_raw.csv")
rows = [{"dataset": d, "shape": s, "repetitions": len(g), "recompute_p50_ms": float(np.median(g.recompute_ms)),
         "recompute_p95_ms": qtl(g.recompute_ms, 0.95), "verify_p50_ms": float(np.median(g.verify_ms)), "verify_p95_ms": qtl(g.verify_ms, 0.95),
         "recomputed_policy_violations": g.recomputed_policy_violations.iloc[0], "recomputed_policy_max_residual": g.recomputed_policy_max_residual.iloc[0]}
        for (d, s), g in rc.groupby(["dataset", "shape"])]
compare("s10 recompute_summary", pd.DataFrame(rows), rd(O / "s10_recompute_summary.csv"), ["dataset", "shape"])

# ---------- s11 history ----------
hi_ = rd(O / "s11_history_raw.csv")
for c in ["released", "fallback_used", "unsafe_release"]:
    hi_[c] = b(hi_[c])
rows = []
for (d, v, p, s), g in hi_.groupby(["dataset", "variant", "proposal", "history_size"]):
    rows.append({"dataset": d, "variant": v, "proposal": p, "history_size": s, "buyers": len(g),
                 "derivable_median": float(np.median(g.derivable_answers)), "remaining_median": float(np.median(g.remaining_answers)),
                 "residual_rows_median": float(np.median(g.residual_rows)), "build_p50_ms": float(np.median(g.snapshot_build_ms)),
                 "build_p95_ms": qtl(g.snapshot_build_ms, 0.95), "violations_before_median": float(np.median(g.violations_before)),
                 "released_rate": g.released.mean(), "fallback_rate": g.fallback_used.mean(), "unsafe_releases": int(g.unsafe_release.sum()),
                 "sweeps_max": g.sweeps.max(), "gate_p50_ms": float(np.median(g.gate_ms)), "gate_p95_ms": qtl(g.gate_ms, 0.95),
                 "assets_price_increased_median": float(np.median(g.assets_price_increased)),
                 "history_target_reduction_median": float(np.median(g.history_target_reduction_median))})
compare("s11 history_summary", pd.DataFrame(rows), rd(O / "s11_history_summary.csv"), ["dataset", "variant", "proposal", "history_size"])
dg = rd(O / "s11_history_pinned_diagnostic.csv")
dg["active_repair_verified"] = b(dg["active_repair_verified"])
rows = [{"dataset": d, "proposal": p, "history_size": s, "buyers": len(g), "active_repair_verified_rate": g.active_repair_verified.mean(),
         "sweeps_median": float(np.median(g.sweeps)), "max_residual_after_median": float(np.median(g.max_residual_after))}
        for (d, p, s), g in dg.groupby(["dataset", "proposal", "history_size"])]
compare("s11 pinned_diagnostic_summary", pd.DataFrame(rows), rd(O / "s11_history_pinned_diagnostic_summary.csv"), ["dataset", "proposal", "history_size"])

# ---------- s12 protocol ----------
rows = []
for name in ["v4_faults", "v4_concurrency", "closure_faults", "closure_concurrency", "closure_delivery"]:
    f = rd(O / f"s12_{name}.csv")
    p = b(f["passed"])
    rows.append({"campaign": name, "rows": len(f), "passed": int(p.sum()), "pass_rate": float(p.mean())})
compare("s12 protocol_summary", pd.DataFrame(rows), rd(O / "s12_protocol_summary.csv"), ["campaign"])

# ---------- sweep budget curve ----------
fams = [(f"s01 natural proposals, tolerance {t:.0e}", g) for t, g in nat.groupby("tolerance")]
fams.append(("s08 declaration coverage", cov))
fams += [(f"s09 pricing-shape proposals, {k}", g) for k, g in sg.groupby("shape")]
fams += [(f"s11 buyer history, {k}", g) for k, g in hi_.groupby("variant")]
rows = []
for label, g in fams:
    s = g.sweeps_to_verify.to_numpy(float); w = b(g.verified_within_cap).to_numpy()
    r = {"family": label, "gate_calls": len(g), "sweeps_p50": float(np.median(s)), "sweeps_p95": qtl(s, 0.95), "sweeps_max": float(s.max()),
         "not_verified_within_cap": int((~w).sum()), "cap": int(cfg["sweeps_to_verify_cap"])}
    for bud in [80, 160, 400, 1000]:
        r[f"share_needing_more_than_{bud}"] = float(np.mean((s > bud) | ~w))
    rows.append(r)
compare("final_sweep_budget_curve", pd.DataFrame(rows), rd(O / "final_sweep_budget_curve.csv"), ["family"])

# ---------- claim-audit values ----------
unsafe_total = 0
for f in sorted(O.glob("*.csv")):
    d = rd(f)
    if "unsafe_release" in d.columns:
        unsafe_total += int(b(d["unsafe_release"]).sum())
p12 = nat[nat.tolerance == 1e-12]; p7 = nat[nat.tolerance == 1e-7]
ss = rd(O / "s04_scale_summary.csv")
claims = {
    "unsafe releases across all raw tables": unsafe_total,
    "1e-12 released/fallback/max sweeps": (int(p12.released.sum()), int(p12.fallback_used.sum()), int(p12.sweeps.max())),
    "1e-7 releases with saving >1% (max)": (int((p7.n_saving_gt_1pct > 0).sum()), round(float(p7.max_saving_vs_list.max()), 4)),
    "1e-12 releases with saving >0.1% (max)": (int((p12.n_saving_gt_0p1pct > 0).sum()), float(p12.max_saving_vs_list.max())),
    "attacks released": int(att.released.sum()),
    "OSQP 1e-12 verified min; v4-stop min": (float(ss[ss.method == "osqp_qp"].verified_rate.min()), float(ss[ss.method == "osqp_qp_v4_stop"].verified_rate.min())),
    "partial encodings with complete violations": (int(((cov.groupby(["dataset", "encoding"]).complete_violations_after.max() > 0)).sum()),
                                                    int((cov.groupby(["dataset", "encoding"]).encoded_share.first() < 1).sum())),
    "infeasible shape policies": int((~b(rd(O / "s09_shape_policy.csv").feasible_at_1e12)).sum()),
    "history released/fallback/max sweeps": (int(hi_.released.sum()), int(hi_.fallback_used.sum()), int(hi_.sweeps_to_verify.max()), int(hi_.sweeps.max())),
    "history max sweeps by variant": hi_.groupby("variant").sweeps.max().to_dict(),
}
for k, v in claims.items():
    print(f"    claim value: {k}: {v}")
print(f"SUMMARY: {sum(ok for _, ok in results)}/{len(results)} recomputations match")
