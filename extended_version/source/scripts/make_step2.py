"""Facts and tables of the extension campaign (Notebook 08) for the Arbiter PVLDB paper.

Usage: python3 -I make_step2.py NB08_RUN_DIR PAPER_DIR

Every number is read from a CSV of the Notebook 08 evidence package; nothing is
typed by hand. The script writes scripts/facts_step2.json (the numbers the prose
cites, with their source file) and the extended-version tables
extended/tab_ext_*.tex.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RUN, PAPER = map(Path, sys.argv[1:3])
O = RUN / "outputs"
X = PAPER / "extended"
facts = {}
DS = ["NYC Taxi 2024-01", "TPC-H SF1", "UCI Adult"]
SHORT = {"NYC Taxi 2024-01": "NYC Taxi", "TPC-H SF1": "TPC-H SF1", "UCI Adult": "UCI Adult", "combined": "Combined"}


def rd(name):
    return pd.read_csv(O / name, float_precision="round_trip")


def flag(series):
    return series.map(lambda v: str(v).strip().lower() in {"true", "1", "1.0"})


def fact(key, value, source):
    def clean(v):
        if isinstance(v, dict):
            return {str(k): clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (np.bool_,)):
            return bool(v)
        return v
    facts[key] = {"value": clean(value), "source": source}


def gi(n):
    return f"{int(round(n)):,}".replace(",", "{,}")


def ms(x):
    if x < 10:
        return f"{x:.2f}"
    if x < 100:
        return f"{x:.1f}"
    return gi(x)


def sp(x):
    if x >= 1000:
        return gi(x)
    if x >= 100:
        return f"{x:.0f}"
    if x >= 10:
        return f"{x:.1f}"
    return f"{x:.2f}"


def dollars(v):
    """Dollar amounts: two significant digits below one dollar, cents above."""
    return f"{v:.2g}" if v < 1 else f"{v:,.2f}".replace(",", "{,}")


def percent(fraction, digits):
    """A fraction as a percentage; scientific notation below 0.001 percent."""
    v = 100.0 * float(fraction)
    if v >= 1e-3:
        return f"{v:.{digits}g}\\%"
    exponent = int(math.floor(math.log10(v)))
    mantissa = v / 10 ** exponent
    if round(mantissa, digits - 1) >= 10:
        mantissa, exponent = mantissa / 10, exponent + 1
    m = f"{mantissa:.{digits}g}"
    return (f"$10^{{{exponent}}}$\\%" if m == "1" else f"${m} \\times 10^{{{exponent}}}$\\%")


# ---------------------------------------------------------------- environment
env = json.loads((RUN / "environment.json").read_text())
quality = pd.read_csv(RUN / "stage_quality_checks.csv")
prov = pd.read_csv(RUN / "part_provenance.csv")
fact("env", {"cpu": env["cpu_model"], "cpus": env["logical_cpu_count"], "memory_gb": env["memory_bytes"] / 1e9,
             "python": env["python"], "packages": env["packages"], "notebook": env["notebook"]}, "environment.json")
fact("run", {"parts": int(len(prov)), "valid_parts": int(flag(prov["valid_for_this_run"]).sum()),
             "stage_seconds": float(prov["elapsed_s"].sum()), "sessions": int(prov["session_id"].nunique()),
             "checks": int(len(quality)), "fatal_checks": int((quality["severity"] == "fatal").sum()),
             "failed_checks": int((~flag(quality["passed"])).sum())}, "part_provenance.csv; stage_quality_checks.csv")

# ---------------------------------------------------------------- s21 GROUP BY
s21 = rd("s21_summary.csv").set_index("dataset")
families = ["grouped_partition_union", "partition_union_new_assets", "group_rollup", "group_slice", "group_filter"]
fact("s21_assets", {"new_total": int(s21["new_assets"].sum()), "grouped_total": int(s21["new_grouped_assets"].sum()),
                    "new_by_dataset": s21["new_assets"].to_dict()}, "s21_summary.csv")
fact("s21_relations", {"total": int(s21["new_relations"].sum()),
                       "by_family": {f: int(s21[f"relations_{f}"].sum()) for f in families}}, "s21_summary.csv")
metrics = rd("s21_compiler_metrics.csv")
fact("s21_oracle", {"precision_min": float(metrics["precision"].min()), "recall_min": float(metrics["recall"].min()),
                    "oracle_relations": int(metrics["oracle_relations"].sum())}, "s21_compiler_metrics.csv")
checks = rd("s21_executed_checks.csv")
fact("s21_executed", {"checks": int(len(checks)), "passed": int(flag(checks["passed"]).sum()),
                      "per_family_dataset": int(checks.groupby(["dataset", "family"]).size().min())}, "s21_executed_checks.csv")
neg = rd("s21_negative_audit.csv")
fact("s21_negatives", {"total": int(len(neg)), "now_supported": int(flag(neg["now_supported"]).sum()),
                       "still_rejected": int((~flag(neg["now_supported"])).sum()), "passed": int(flag(neg["passed"]).sum())},
     "s21_negative_audit.csv")
fact("s21_base", {"relations_unchanged": bool(flag(s21["base_relations_unchanged"]).all()),
                  "prices_unchanged": bool(flag(s21["base_prices_reproduced"]).all()),
                  "parse_unchanged": bool(flag(s21["base_parse_matches_construction"]).all())}, "s21_summary.csv")
self_test = rd("s21_parser_self_test.csv")
fact("s21_parser_self_test", {"cases": int(len(self_test)), "passed": int(flag(self_test["passed"]).sum())}, "s21_parser_self_test.csv")
pg = rd("s21_proposals_gate.csv")
fact("s21_proposals", {"count": int(len(pg)), "released": int(flag(pg["released"]).sum()),
                       "fallback": int(flag(pg["fallback_used"]).sum()), "unsafe": int(flag(pg["unsafe_release"]).sum()),
                       "violations_min": int(pg["violations_before"].min()), "violations_max": int(pg["violations_before"].max()),
                       "new_family_rows_max": int(pg["new_family_rows_violated_before"].max()),
                       "sweeps_max": int(pg["sweeps"].max()), "gate_p50_ms": float(pg["runtime_ms"].median()),
                       "max_residual_after": float(pg["max_residual_after"].max())}, "s21_proposals_gate.csv")
pooled = rd("s21_model_pooled.csv")
fact("s21_model_r2", pooled.groupby("model")["pooled_r2"].agg(["min", "max"]).to_dict(orient="index"), "s21_model_pooled.csv")
at = rd("s21_attacks.csv")
fact("s21_attacks", {"count": int(len(at)), "released": int(flag(at["released"]).sum()),
                     "fallback": int(flag(at["fallback_used"]).sum()), "unsafe": int(flag(at["unsafe_release"]).sum()),
                     "sweeps_max": int(at["sweeps"].max()), "p50_ms": float(at["runtime_ms"].median()),
                     "p95_ms": float(at["runtime_ms"].quantile(0.95)), "violations_min": int(at["violations_before"].min()),
                     "violations_max": int(at["violations_before"].max()),
                     "max_residual_after": float(at["max_residual_after"].max()),
                     "by_family": at.groupby("family").size().to_dict()}, "s21_attacks.csv")

# ---------------------------------------------------------------- s22 audit
s22 = rd("s22_summary.csv").set_index("dataset")
a22 = rd("s22_attacks.csv")
am = rd("s22_audit_mode.csv")
fact("s22", {"targets": int(s22["wide_targets"].sum()), "r1_rows": int(s22["r1_rows_on_wide_targets"].sum()),
             "list_flags": int(s22["list_price_flags"].sum()),
             "interval_assets_min": int(s22["interval_targets_audited"].min()), "interval_assets_max": int(s22["interval_targets_audited"].max()),
             "audit_p50_ms_min": float(s22["audit_all_targets_p50_ms"].min()), "audit_p50_ms_max": float(s22["audit_all_targets_p50_ms"].max()),
             "attacks": int(len(a22)), "released_first": int(flag(a22["released_first"]).sum()),
             "detected": int(flag(a22["audit_detected"]).sum()),
             "parts_min": int(a22["chain_parts"].min()), "parts_max": int(a22["chain_parts"].max()),
             "rounds_min": int(a22["enforcement_rounds"].min()), "rounds_max": int(a22["enforcement_rounds"].max()),
             "rounds_median": float(a22["enforcement_rounds"].median()),
             "rounds_counts": a22["enforcement_rounds"].value_counts().sort_index().to_dict(),
             "flagged_after": int(flag(a22["flagged_after"]).sum()),
             "unsafe": int(flag(a22["unsafe_first"]).sum() + flag(a22["unsafe_final"]).sum()),
             "total_ms_p50": float(a22["total_ms"].median()), "total_ms_max": float(a22["total_ms"].max()),
             "first_gate_ms_p50": float(a22["first_gate_ms"].median()),
             "audit_mode_releases": int(len(am)), "audit_mode_flagged": int(flag(am["audit_flags_undeclared_violation"]).sum()),
             "audit_mode_flagged_D": int(flag(am[am.declaration == "D"]["audit_flags_undeclared_violation"]).sum()),
             "audit_mode_D": int((am.declaration == "D").sum()),
             "audit_mode_flagged_PF": int(flag(am[am.declaration == "PF"]["audit_flags_undeclared_violation"]).sum()),
             "audit_mode_PF": int((am.declaration == "PF").sum()),
             "audit_mode_p50_ms_min": float(am.groupby("dataset")["audit_p50_ms"].median().min()),
             "audit_mode_p50_ms_max": float(am.groupby("dataset")["audit_p50_ms"].median().max())},
     "s22_summary.csv; s22_attacks.csv; s22_audit_mode.csv")

# ---------------------------------------------------------------- s23 floors
s23 = rd("s23_summary.csv")
policy = rd("s23_policy.csv")
cells = {}
for row in s23.itertuples(index=False):
    cells[f"{row.floor:g}|{row.tolerance:g}"] = {
        "released": int(row.released), "gate_calls": int(row.gate_calls), "fallback": int(row.fallback),
        "max_saving": float(row.max_saving_vs_list), "gt_1pct": int(row.releases_gt_1pct), "gt_0p1pct": int(row.releases_gt_0p1pct),
        "bound": float(row.corollary_bound), "bound_holds": bool(row.bound_holds), "target_list_min": float(row.target_list_min),
        "sweeps_max": int(row.sweeps_max), "tolerance_usd": float(row.tolerance_usd), "target_list_min_usd": float(row.target_list_min_usd)}
fact("s23", {"cells": cells, "all_bounds_hold": bool(flag(s23["bound_holds"]).all()),
             "all_released": int(s23["released"].sum()), "all_calls": int(s23["gate_calls"].sum()),
             "fallback": int(s23["fallback"].sum()), "policy_verifies": bool(flag(policy["verifies"]).all()),
             "assets_floored": policy.set_index(["dataset", "floor"])["assets_floored"].to_dict()},
     "s23_summary.csv; s23_policy.csv")

# ---------------------------------------------------------------- s24 projection
dist = rd("s24_distances.csv")
cert = rd("s24_certificates.csv")
raw24 = rd("s24_raw.csv")
paired24 = rd("s24_paired.csv")
summary = {}
for method, group in dist.groupby("method"):
    summary[method] = {"l2_low_min": float(group["l2_ratio_to_projection_low"].min()),
                       "l2_high_max": float(group["l2_ratio_to_projection_high"].max()),
                       "l2_median": float(group["l2_ratio_to_projection_high"].median()),
                       "l1_osqp_median": float(group["l1_ratio_to_osqp"].median()),
                       "l1_osqp_min": float(group["l1_ratio_to_osqp"].min()),
                       "l1_osqp_max": float(group["l1_ratio_to_osqp"].max()),
                       "distance_high_median": float(group["distance_to_projection_high"].median()),
                       "verified": int(flag(group["verified"]).sum()), "proposals": int(len(group)),
                       "ratio_above_1p001": int((group["l2_ratio_to_projection_low"] > 1.001).sum())}
status = raw24[raw24.method.str.startswith("clarabel")].groupby(["method", "status"]).size()
timing = raw24.groupby(["dataset", "method"])["runtime_ms"].median().unstack()
speed = paired24[paired24.subset == "all_trials"].set_index(["dataset", "baseline"])
fact("s24", {"methods": summary,
             "bracket_width_max": float(cert["relative_width"].max()), "bracket_width_median": float(cert["relative_width"].median()),
             "feasible_point_bound_max": float(cert["feasible_point_distance_bound"].max()),
             "projection_distance_min": float(cert["projection_distance_lower"].min()),
             "projection_distance_max": float(cert["projection_distance_upper"].max()),
             "certificate_seconds_max": float(cert["certificate_seconds"].max()),
             "clarabel_status": {f"{m}|{s}": int(n) for (m, s), n in status.items()},
             "osqp_iteration_cap": int((raw24[raw24.method == "osqp_qp"]["sweeps"] >= 20000).sum()),
             "osqp_trials": int((raw24.method == "osqp_qp").sum()),
             "timing_p50_ms": {d: timing.loc[d].to_dict() for d in DS},
             "speedups": {f"{d}|{bl}": {"median": float(r.speedup_median), "low": float(r.speedup_ci95_low), "high": float(r.speedup_ci95_high)}
                          for (d, bl), r in speed.iterrows()}},
     "s24_distances.csv; s24_certificates.csv; s24_raw.csv; s24_paired.csv")

# ---------------------------------------------------------------- s25 scale-out
work = rd("s25_workloads.csv")
msum = rd("s25_method_summary.csv")
paired25 = rd("s25_paired.csv")
repro = work[flag(work["compared_with_audited"])]
fact("s25_workloads", {"reproduced": bool(flag(repro["reproduces_audited_workload"]).all()
                                          and flag(repro["reproduces_audited_relations"]).all()
                                          and flag(repro["same_snapshot_digest_as_audited"]).all()),
                       "reproduced_count": int(len(repro)),
                       "oracle_min": float(min(work["oracle_precision"].min(), work["oracle_recall"].min())),
                       "rows": work.set_index(work["dataset"] + "|" + work["size"].astype(str))[["assets", "relations", "rows", "nonzeros", "build_p50_ms", "numeric_bytes"]].to_dict(orient="index")},
     "s25_workloads.csv")
arb = msum[msum.method == "arbiter_active"].set_index("workload")
fact("s25_arbiter", arb[["assets", "runtime_p50_ms", "runtime_p95_ms", "runtime_p99_ms", "verified_rate", "fallback_rate", "sweeps_max"]].to_dict(orient="index"),
     "s25_method_summary.csv")
base = msum[msum.method != "arbiter_active"].set_index(["workload", "method"])
fact("s25_baselines", {f"{w}|{m}": {"p50": float(r.runtime_p50_ms), "p95": float(r.runtime_p95_ms), "verified_rate": float(r.verified_rate)}
                       for (w, m), r in base.iterrows()}, "s25_method_summary.csv")
sp25 = paired25[paired25.subset == "all_trials"].set_index(["workload", "baseline"])
fact("s25_speedups", {f"{w}|{bl}": {"median": float(r.speedup_median), "low": float(r.speedup_ci95_low), "high": float(r.speedup_ci95_high),
                                     "baseline_verified_rate": float(r.baseline_verified_rate)} for (w, bl), r in sp25.iterrows()},
     "s25_paired.csv")
viol = rd("s25_raw.csv").groupby("workload")["violations_before"].agg(["min", "median", "max"])
fact("s25_violations", viol.to_dict(orient="index"), "s25_raw.csv")

# ---------------------------------------------------------------- facts for the prose (added for the paper text)
dummy = cert[cert.model == "dummy_median"].groupby("dataset")["projection_distance_upper"].max()
fact("s24_median_predictor_projection_distance", {d: float(dummy[d]) for d in dummy.index},
     "s24_certificates.csv (upper bound on ||p* - p0|| for the median-predictor proposals)")
cd = dist[(dist.method == "clarabel_default") & (dist.l2_ratio_to_projection_low > 1.001)]
fact("s24_clarabel_default_far", {"proposals": int(len(cd)), "models": sorted(set(cd.model)),
                                  "datasets": sorted(set(cd.dataset)),
                                  "ratio_by_dataset": cd.groupby("dataset")["l2_ratio_to_projection_low"].min().to_dict()},
     "s24_distances.csv (Clarabel at its default tolerance, L2 ratio above 1.001)")
fact("s24_osqp_runtime_max_ms", float(raw24[raw24.method == "osqp_qp"]["runtime_ms"].max()), "s24_raw.csv (OSQP at eps_abs 1e-12)")
fact("s24_clarabel_iterations_max", raw24[raw24.method.str.startswith("clarabel")].groupby("method")["sweeps"].max().to_dict(),
     "s24_raw.csv (Clarabel iteration counts; the iteration limit is 400)")
fact("s24_osqp_distance_max", float(dist[dist.method == "osqp_qp"]["distance_to_projection_high"].max()),
     "s24_distances.csv (upper bound on the distance from the OSQP solution to p*)")
fact("s24_all_trials_verified", {m: float(flag(g["verified"]).mean()) for m, g in raw24.groupby("method")}, "s24_raw.csv")
fact("s23_dollars_per_unit", float((s23["tolerance_usd"] / s23["tolerance"]).median()), "s23_summary.csv (tolerance_usd / tolerance)")
by_model = {}
for method in ["arbiter_active", "full_cyclic"]:
    g = dist[dist.method == method].groupby("model")
    by_model[method] = {m: {"low_min": float(x["l2_ratio_to_projection_low"].min()),
                            "high_median": float(x["l2_ratio_to_projection_high"].median()),
                            "high_max": float(x["l2_ratio_to_projection_high"].max())} for m, x in g}
fact("s24_l2_ratio_by_model", by_model, "s24_distances.csv (L2 change ratio to the exact projection, per model family)")
fact("s24_certificate_sources", {"dual": cert["dual_source"].fillna("").astype(str).value_counts().to_dict(),
                                 "feasible": cert["feasible_source"].fillna("").astype(str).value_counts().to_dict()},
     "s24_certificates.csv (which multipliers and which feasible point gave each bound)")
fact("s21_gate_p50_by_dataset", pg.groupby("dataset")["runtime_ms"].median().to_dict(), "s21_proposals_gate.csv")
fact("s21_violations_by_model", pg.groupby("model")["violations_before"].agg(["min", "max"]).to_dict(orient="index"),
     "s21_proposals_gate.csv")
fact("s23_assets_floored", policy.groupby("floor")["assets_floored"].sum().to_dict(), "s23_policy.csv (list prices raised by each floor, summed over datasets)")

# ---------------------------------------------------------------- paper table: proposal-scale comparison (s24)
msum24 = rd("s24_method_summary.csv").set_index(["dataset", "method"])
methods_tab = [("arbiter_active", "Arbiter"), ("full_cyclic", "Full cyclic"), ("osqp_qp", r"OSQP, $10^{-12}$"),
               ("clarabel_default", r"Clarabel, $10^{-8}$"), ("clarabel_tight", r"Clarabel, $10^{-12}$")]
rows_tab = []
for method, label in methods_tab:
    cells_ = [label]
    for d in DS:
        r = msum24.loc[(d, method)]
        assert float(r.verified_rate) == 1.0, (d, method)
        cells_.append(f"{ms(r.runtime_p50_ms)} / {ms(r.runtime_p95_ms)}")
    cells_.append(f"{summary[method]['l2_median']:.2f}")
    rows_tab.append(" & ".join(cells_) + r" \\")
tight_ip = int(status.get(("clarabel_tight", "InsufficientProgress"), 0))
tight_n = int(status.xs("clarabel_tight", level=0).sum())
fact("s24_clarabel_tight_insufficient_progress", [tight_ip, tight_n], "s24_raw.csv (status of Clarabel at 1e-12)")
(X / "tab_ext_projection.tex").write_text("\n".join([
    r"\begin{table}[h]", r"\centering",
    r"\caption{Model proposals in the extension campaign at tolerance $10^{-12}$: median / p95 latency in ms over 3 timed repetitions per proposal, and L2: median ratio of the L2 price change to that of the exact projection $\pvec^{\star}$. Clarabel rows give the solver tolerance. Every output passed verification.}",
    r"\label{tab:ext-projection}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
    r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
    r"Method & NYC Taxi & TPC-H SF1 & UCI Adult & L2 \\", r"\midrule",
    *rows_tab, r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]), encoding="utf-8")


# ---------------------------------------------------------------- extended-version tables
def workload_label(w):
    ds, size = w.split("|")
    if size.startswith("3x"):
        return r"Combined 3$\times$" + gi(int(size[2:]))
    return f"{SHORT[ds]} {gi(int(size))}"


order = ["NYC Taxi 2024-01|4096", "NYC Taxi 2024-01|8192", "TPC-H SF1|4096", "TPC-H SF1|8192", "TPC-H SF1|12288",
         "UCI Adult|4096", "UCI Adult|8192", "UCI Adult|12288", "combined|3x4096", "combined|3x8192"]
lines = []
for w in order:
    a = arb.loc[w]
    wrow = work[(work["dataset"] + "|" + work["size"].astype(str)) == w].iloc[0]
    cells_ = [workload_label(w), gi(a.assets), gi(wrow["relations"]), gi(wrow["rows"]), ms(wrow["build_p50_ms"]),
              ms(a.runtime_p50_ms), ms(a.runtime_p95_ms)]
    for bl in ["full_cyclic", "osqp_qp", "clarabel_default"]:
        r = sp25.loc[(w, bl)]
        assert float(r.baseline_verified_rate) == 1.0, (w, bl)
        cells_.append(f"{sp(r.speedup_median)} [{sp(r.speedup_ci95_low)}, {sp(r.speedup_ci95_high)}]")
    lines.append(" & ".join(cells_) + r" \\")
(X / "tab_ext_scale.tex").write_text("\n".join([
    r"\begin{table}[h]", r"\centering",
    r"\caption{Scale-out in the extension campaign at tolerance $10^{-12}$ on exact per-row attack candidates (12 per dataset workload, 18 per combined workload; budget 25\%). Build: snapshot build time in ms (median of three builds; one build for combined workloads). p50 and p95: Arbiter's latency in ms over 10 timed repetitions after 3 warm-ups. Speedups: paired, with 95\% bootstrap intervals, over full cyclic repair, OSQP at $\epsilon_{\mathrm{abs}} = 10^{-12}$, and Clarabel at its default tolerance, each timed over 3 repetitions after one warm-up. Every output of every method passed verification. Combined workloads join the three datasets' snapshots at 4{,}096 and 8{,}192 assets each.}",
    r"\label{tab:ext-scale}", r"\footnotesize", r"\setlength{\tabcolsep}{2.5pt}",
    r"\begin{tabular}{@{}lrrrrrrlll@{}}", r"\toprule",
    r"Workload & Assets & Relations & Rows & Build & p50 & p95 & Full cyclic & OSQP & Clarabel \\", r"\midrule",
    *lines, r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]), encoding="utf-8")

fam_label = {"grouped_partition_union": "Grouped partition union (R1)", "partition_union_new_assets": "Partition union, new assets (R1)",
             "group_rollup": "Roll-up (R4)", "group_slice": "Slice (R5)", "group_filter": "Group filter (R6)"}
rel_rows = []
for f in families:
    m = metrics[metrics.family == f]
    a = at[at.family == f]
    rel_rows.append(" & ".join([fam_label[f], gi(m["compiled_relations"].sum()), gi(m["oracle_relations"].sum()),
                                f"{m['precision'].min():.3f}", f"{m['recall'].min():.3f}",
                                f"{int(flag(checks[checks.family == f]['passed']).sum())}/{int((checks.family == f).sum())}",
                                f"{int(flag(a['released']).sum())}/{len(a)}", str(int(a["sweeps"].max()))]) + r" \\")
(X / "tab_ext_groupby.tex").write_text("\n".join([
    r"\begin{table}[h]", r"\centering",
    r"\caption{GROUP BY extension in the extension campaign, summed over the three datasets: compiled and oracle relations per rule family, the smallest per-dataset precision and recall, executed checks passed in DuckDB, exact per-row attacks released at tolerance $10^{-12}$ with no fallback and no unverified release, and the largest sweep count of those releases.}",
    r"\label{tab:ext-groupby}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
    r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
    r"Family & Compiled & Oracle & Prec. & Rec. & Checks & Attacks & Sweeps \\", r"\midrule",
    *rel_rows, r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]), encoding="utf-8")

floor_rows = []
for row in s23.sort_values(["floor", "tolerance"], ascending=[True, False]).itertuples(index=False):
    floor_rows.append(" & ".join([
        f"$10^{{{int(round(math.log10(row.floor)))}}}$", "USD~" + dollars(row.target_list_min_usd),
        f"$10^{{{int(round(math.log10(row.tolerance)))}}}$",
        f"{int(row.released)}/{int(row.gate_calls)}", str(int(row.sweeps_max)),
        f"{int(row.releases_gt_1pct)}/{int(row.released)}", f"{int(row.releases_gt_0p1pct)}/{int(row.released)}",
        percent(row.max_saving_vs_list, 4), percent(row.corollary_bound, 3)]) + r" \\")
(X / "tab_ext_floors.tex").write_text("\n".join([
    r"\begin{table}[h]", r"\centering",
    r"\caption{Price floors in the extension campaign: 45 retrained proposals per cell, gated against list prices $\max(\ell_i, f)$. Dollars set one price unit to USD~2{,}176. Threshold columns count releases that keep a saving above 1\% or 0.1\% of the target's floored list price. Max: largest saving. Bound: $\varepsilon / \ell_{\min}$ of Corollary 6.4. No cell used fallback.}",
    r"\label{tab:ext-floors}", r"\footnotesize", r"\setlength{\tabcolsep}{3.5pt}",
    r"\begin{tabular}{@{}lllrrrrrr@{}}", r"\toprule",
    r"Floor $f$ & Dollars & $\varepsilon$ & Released & Sweeps & ${>}1\%$ & ${>}0.1\%$ & Max & Bound \\", r"\midrule",
    *floor_rows, r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]), encoding="utf-8")

chain_rows = []
for d in DS:
    s = s22.loc[d]
    a = a22[a22.dataset == d]
    chain_rows.append(" & ".join([SHORT[d], str(int(s.wide_targets)), str(int(s.r1_rows_on_wide_targets)), gi(s.interval_targets_audited),
                                  ms(s.audit_all_targets_p50_ms), f"{int(flag(a['audit_detected']).sum())}/{len(a)}",
                                  f"{int(a['chain_parts'].min())}--{int(a['chain_parts'].max())}",
                                  f"{int(a['enforcement_rounds'].min())}--{int(a['enforcement_rounds'].max())}",
                                  f"{ms(a['total_ms'].median())} / {ms(a['total_ms'].max())}"]) + r" \\")
(X / "tab_ext_chains.tex").write_text("\n".join([
    r"\begin{table}[h]", r"\centering",
    r"\caption{Chain audit in the extension campaign. Targets: interval assets that only chains of existing intervals tile; R1 rows: decompositions that R1 emits for them. Audited: interval assets in one audit pass, with its median time in ms. Detected: attacks (budgets 10\%, 25\%, 50\%) whose violated chain the audit flags. Parts: parts of the cheapest flagged chain. Rounds: lazy-enforcement rounds until the audit flags nothing. Time: median / largest total time per attack in ms.}",
    r"\label{tab:ext-chains}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
    r"\begin{tabular}{@{}lrrrrrrrr@{}}", r"\toprule",
    r"Dataset & Targets & R1 rows & Audited & Audit & Detected & Parts & Rounds & Time \\", r"\midrule",
    *chain_rows, r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]), encoding="utf-8")

(PAPER / "scripts" / "facts_step2.json").write_text(json.dumps(facts, indent=2, sort_keys=True), encoding="utf-8")
print(len(facts), "facts written")
