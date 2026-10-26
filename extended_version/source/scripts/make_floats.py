"""Write every data table and data figure of the Arbiter PVLDB paper.

Usage: python3 -I make_floats.py NB07_RUN_DIR AUDITED_V4_DIR CLOSURE_DIR HARDENING_DIR PAPER_DIR

Inputs are the extracted evidence packages. Every number in the generated
tables is read from a CSV in those packages; nothing is typed by hand. The
script also writes scripts/facts.json, the numbers the prose cites, with their source.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RUN, V4, CL, HD, PAPER = map(Path, sys.argv[1:6])
O = RUN / "outputs"
T = PAPER / "floats" / "tables"
F = PAPER / "floats" / "figures"
A = PAPER / "assets"
facts = {}

DS = ["NYC Taxi 2024-01", "TPC-H SF1", "UCI Adult"]
SHORT = {"NYC Taxi 2024-01": "NYC Taxi", "TPC-H SF1": "TPC-H SF1", "UCI Adult": "UCI Adult"}


def rd(path):
    return pd.read_csv(path, float_precision="round_trip")


def b(series):
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])


def gi(n):
    """Integer with LaTeX thousands separators."""
    s = f"{int(round(n)):,}"
    return s.replace(",", "{,}")


def ms(x):
    """Milliseconds: two decimals below 10, one below 100, grouped integer above."""
    if x < 10:
        return f"{x:.2f}"
    if x < 100:
        return f"{x:.1f}"
    return gi(x)


def sp(x):
    """Speedup with three significant digits."""
    if x >= 100:
        return f"{x:.0f}"
    if x >= 10:
        return f"{x:.1f}"
    return f"{x:.2f}"


def pct(x, digits=1):
    return f"{100 * x:.{digits}f}\\%"


def fact(key, value, source):
    if isinstance(value, (np.integer,)):
        value = int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    facts[key] = {"value": value, "source": source}


def write(name, text):
    (T / f"{name}.tex").write_text(text.strip() + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# Table 1: datasets and compiled workload (audited v4, tolerance-independent)
# --------------------------------------------------------------------------
meta = rd(V4 / "00_workload/dataset_metadata.csv").set_index("dataset")
cm = rd(V4 / "00_workload/compiler_metrics.csv").set_index("dataset")
rel = rd(V4 / "00_workload/compiled_relations.csv")
relc = rel.groupby(["dataset", "relation_type"]).size().unstack()
specs = json.loads((V4 / "00_workload/table_specs.json").read_text())
build = rd(O / "s04_build_summary.csv")
b4096 = build[build.scale == 4096].set_index("dataset")
# Base rows count the rows of the tables the workload reads (table_specs.json). For
# NYC Taxi and UCI Adult that is the whole dataset; TPC-H SF1 has eight tables, of
# which the workload reads lineitem, orders, and customer (corrected 2026-10-24:
# the earlier build printed the row count of all eight tables).
base_rows = {d: sum(int(t["row_count"]) for t in specs[d]) for d in DS}
for d in ("NYC Taxi 2024-01", "UCI Adult"):
    assert base_rows[d] == int(meta.loc[d, "total_rows"]), d
fact("tpch_rows_all_tables", int(meta.loc["TPC-H SF1", "total_rows"]), "v4 00_workload/dataset_metadata.csv (all eight TPC-H tables)")
rows = []
for d in DS:
    # Single-column layout (2026-10-24): the table count, the supported-asset count, and the
    # nonzeros move to the caption, so the table fits one column.
    assert int(cm.loc[d, "expected_supported_assets"]) == 4096, d
    rows.append(
        f"{SHORT[d]} & {gi(base_rows[d])} & "
        f"{gi(cm.loc[d, 'assets'] - cm.loc[d, 'expected_supported_assets'])} & "
        f"{gi(relc.loc[d, 'decomposition'])} & {gi(relc.loc[d, 'projection_derivation'])} & {gi(relc.loc[d, 'filter_derivation'])} & "
        f"{gi(b4096.loc[d, 'constraints'])} \\\\"
    )
    fact(f"rows_{SHORT[d]}", int(base_rows[d]), "v4 00_workload/table_specs.json (rows of the tables the workload reads)")
tables_per_ds = {d: len(specs[d]) for d in DS}
assert tables_per_ds == {"NYC Taxi 2024-01": 1, "TPC-H SF1": 3, "UCI Adult": 1}, tables_per_ds
nz = gi(b4096.loc[DS, "nonzeros"].min()) + " to " + gi(b4096.loc[DS, "nonzeros"].max())
write("tab_data", r"""
\begin{table}[t]
\centering
\caption{Datasets and compiled workload, 4{,}096 supported SQL assets each. Base rows: rows of the tables the workload reads (three for TPC-H SF1). Uns.: rejected unsupported queries. Decomp., Proj., Filter: compiled relations by family. Rows: snapshot rows, including one nonnegativity row per asset (""" + nz + r""" nonzeros).}
\label{tab:data}
\footnotesize
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
Dataset & Base rows & Uns. & Decomp. & Proj. & Filter & Rows \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
fact("relations_total", int(len(rel)), "v4 00_workload/compiled_relations.csv")
fact("relations_by_dataset", {SHORT[d]: int(cm.loc[d, "compiled_relations"]) for d in DS}, "v4 compiler_metrics.csv")
fact("unsupported_total", int((cm["assets"] - cm["expected_supported_assets"]).sum()), "v4 compiler_metrics.csv")
fact("assets_total", int(cm["assets"].sum()), "v4 compiler_metrics.csv")
fact("precision_recall", {SHORT[d]: [float(cm.loc[d, "precision"]), float(cm.loc[d, "recall"])] for d in DS}, "v4 compiler_metrics.csv")
sem = rd(V4 / "00_workload/semantic_execution_checks.csv")
fact("semantic_checks", [int(b(sem.passed).sum()), int(len(sem))], "v4 semantic_execution_checks.csv")
ct = rd(V4 / "00_workload/compiler_timing.csv").set_index("dataset")
fact("compile_ms", {SHORT[d]: float(ct.loc[d, "sql_parse_and_relation_compile_ms"]) for d in DS}, "v4 compiler_timing.csv (2 vCPU)")

# --------------------------------------------------------------------------
# Table: tolerance comparison (Notebook 07 s01)
# --------------------------------------------------------------------------
nat = rd(O / "s01_natural_gate.csv")
for c in ["released", "rejected", "fallback_used", "unsafe_release"]:
    nat[c] = b(nat[c])
rows = []
for t, label in [(1e-7, r"$10^{-7}$"), (1e-9, r"$10^{-9}$"), (1e-12, r"$10^{-12}$")]:
    g = nat[nat.tolerance == t]
    n = len(g)
    gt1 = int((g.n_saving_gt_1pct > 0).sum())
    gt1c = int((g.n_conservative_gt_1pct > 0).sum())
    gt01 = int((g.n_saving_gt_0p1pct > 0).sum())
    mx = float(g.max_saving_vs_list.max())
    mx_txt = f"{100 * mx:.1f}\\%" if mx >= 0.01 else (f"{100 * mx:.2f}\\%" if mx >= 1e-4 else f"{100 * mx:.4f}\\%")
    rows.append(
        f"{label} & {int(g.sweeps.max())} & {ms(float(np.median(g.gate_ms)))} & "
        f"{int(g.relations_above_1e12.max())} & {gt1}/{n} & {gt1c}/{n} & {gt01}/{n} & {mx_txt} \\\\"
    )
    key = {1e-7: "1e-7", 1e-9: "1e-9", 1e-12: "1e-12"}[t]
    fact(f"tol_{key}", {"released": int(g.released.sum()), "fallback": int(g.fallback_used.sum()), "sweeps_max": int(g.sweeps.max()),
                        "gate_p50_ms": float(np.median(g.gate_ms)), "gt1": gt1, "gt1_cons": gt1c, "gt01": gt01, "max_saving": mx,
                        "relations_above_1e12_max": int(g.relations_above_1e12.max()),
                        "relations_above_1e12_median": float(np.median(g.relations_above_1e12)),
                        "relations_gt1_total": int(g.n_saving_gt_1pct.sum())}, "nb07 s01_natural_gate.csv")
write("tab_tolerance", r"""
\begin{table}[t]
\centering
\caption{Leftover savings over the 45 model proposals at three tolerances, relative to the target's list price (for (c), the larger of the posted and list prices). Rel.: largest number of relations with residual above $10^{-12}$ in one release. Sweeps/p50 as in Table~\ref{tab:release}.}
\label{tab:tolerance}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lrrrrrrr@{}}
\toprule
 & & & & \multicolumn{3}{c}{Releases with saving above} & \\
\cmidrule(lr){5-7}
$\varepsilon$ & Sweeps & p50 & Rel. & 1\% & 1\% (c) & 0.1\% & Max \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
top = rd(O / "s01_natural_top_relations.csv")
t7 = top[(top.tolerance == 1e-7) & (top["rank"] == 0)]
worst = t7.sort_values("saving_vs_list").iloc[-1]
fact("tol_1e-7_worst_relation", {"dataset": worst.dataset, "type": worst.relation_type, "residual": float(worst.residual),
                                 "target_list_price": float(worst.target_list_price), "saving": float(worst.saving_vs_list)},
     "nb07 s01_natural_top_relations.csv")
fact("tol_1e-7_top_relation_types", t7.relation_type.value_counts().to_dict(), "nb07 s01_natural_top_relations.csv")
pol = rd(V4 / "01_proposals_attacks/controlled_policy.csv", )
fact("list_price_min_max", [float(pol.target_price.min()), float(pol.target_price.max())], "v4 controlled_policy.csv")
fact("list_price_below_1e-7", int((pol.target_price < 1e-7).sum()), "v4 controlled_policy.csv")
fact("list_price_median", {SHORT[d]: float(pol[pol.dataset == d].target_price.median()) for d in DS}, "v4 controlled_policy.csv")
# Smallest list price among relation targets (decomposition targets and derived assets), used by Corollary 6.4.
_tgt = pd.concat([rel[rel.relation_type == "decomposition"][["dataset", "target_id"]].rename(columns={"target_id": "asset_id"}),
                  rel[rel.relation_type != "decomposition"][["dataset", "derived_id"]].rename(columns={"derived_id": "asset_id"})])
_tp = _tgt.merge(pol, on=["dataset", "asset_id"], how="left")
assert _tp.target_price.notna().all()
fact("list_price_min_relation_target", float(_tp.target_price.min()), "v4 controlled_policy.csv joined with v4 compiled_relations.csv")
fact("tolerance_bound_examples", {"theta_0.1pct_max_tol": 1e-3 * float(_tp.target_price.min()),
                                  "bound_at_1e-12_pct": 100 * 1e-12 / float(_tp.target_price.min()),
                                  "bound_at_1e-7_ratio": 1e-7 / float(_tp.target_price.min())}, "derived from list_price_min_relation_target")
aud = rd(V4 / "01_proposals_attacks/natural_proposal_gate.csv")
n7 = nat[nat.tolerance == 1e-7]
m = aud.merge(n7, on=["dataset", "model", "seed"], suffixes=("_v4", "_r"))
same = bool(len(m) == 45 and all((m[f"{c}_v4"] == m[f"{c}_r"]).all() for c in ["violations_before", "max_residual_after", "l1_adjustment"]))
fact("s01_1e-7_equals_audited", same, "nb07 s01 vs v4 natural_proposal_gate.csv")

# --------------------------------------------------------------------------
# Table: release outcomes under untrusted proposals (RQ1 summary)
# --------------------------------------------------------------------------
att = rd(O / "s02_attacks.csv")
for c in ["released", "fallback_used", "unsafe_release"]:
    att[c] = b(att[c])
sg = rd(O / "s09_shape_gate.csv")
for c in ["released", "fallback_used", "unsafe_release"]:
    sg[c] = b(sg[c])
sa = rd(O / "s09_shape_attacks.csv")
for c in ["released", "fallback_used", "unsafe_release"]:
    sa[c] = b(sa[c])
hi = rd(O / "s11_history_raw.csv")
for c in ["released", "fallback_used", "unsafe_release"]:
    hi[c] = b(hi[c])
n12 = nat[nat.tolerance == 1e-12]


def outcome_row(label, frame, time_col):
    return (f"{label} & {gi(len(frame))} & {gi(frame.violations_before.min())}--{gi(frame.violations_before.max())} & "
            f"{gi(frame.released.sum())} & {int(frame.sweeps.max())} & "
            f"{ms(float(np.median(frame[time_col])))} \\\\")


for _frame in (n12, att, sg, sa, hi):
    assert int(_frame.fallback_used.sum()) == 0, "tab_release caption states that no case used the fallback"
rows = [
    outcome_row("Model proposals", n12, "gate_ms"),
    outcome_row("Per-row attacks", att, "repair_ms"),
    outcome_row("Shape proposals", sg, "gate_ms"),
    outcome_row("Shape attacks", sa, "gate_ms"),
    outcome_row("Buyer histories", hi, "gate_ms"),
]
write("tab_release", r"""
\begin{table}[t]
\centering
\caption{Release outcomes at $10^{-12}$; no case used the fallback. Violated: rows violated before the gate; Released: passed independent verification; Sweeps/p50: largest sweep count and median gate time (ms).}
\label{tab:release}
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{@{}lrrrrr@{}}
\toprule
Workload & Cases & Violated & Released & Sweeps & p50 \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
fact("attacks", {"n": int(len(att)), "released": int(att.released.sum()), "fallback": int(att.fallback_used.sum()),
                 "sweeps_max": int(att.sweeps.max()), "viol_min": int(att.violations_before.min()), "viol_max": int(att.violations_before.max()),
                 "viol_max_by_ds": {SHORT[d]: int(att[att.dataset == d].violations_before.max()) for d in DS},
                 "repair_p50": float(np.median(att.repair_ms)), "repair_p95": float(np.quantile(att.repair_ms, 0.95)),
                 "repair_p50_by_ds": {SHORT[d]: float(np.median(att[att.dataset == d].repair_ms)) for d in DS},
                 "post_residual_max": float(att.max_residual_after.max()),
                 "certified_residual_min": float(att.certified_target_residual.min())}, "nb07 s02_attacks.csv")
v4cmp = rd(O / "s02_attack_v4_comparison.csv")
fact("attacks_vs_v4", {c: int(v4cmp[c].sum()) for c in ["cases_1e12", "same_target_rows", "same_release_decision", "same_violations_before", "same_sweeps"]},
     "nb07 s02_attack_v4_comparison.csv")
fact("natural", {"viol_min": int(n12.violations_before.min()), "viol_max": int(n12.violations_before.max()),
                 "sweeps_by_model": n12.groupby("model").sweeps.max().astype(int).to_dict()}, "nb07 s01_natural_gate.csv")
fact("history", {"n": int(len(hi)), "released": int(hi.released.sum()), "fallback": int(hi.fallback_used.sum()),
                 "sweeps_max_by_variant": hi.groupby("variant").sweeps.max().astype(int).to_dict(),
                 "gate_p50": float(np.median(hi.gate_ms))}, "nb07 s11_history_raw.csv")

# --------------------------------------------------------------------------
# Table: proposal-level comparison under heavy load (s01 heavy)
# --------------------------------------------------------------------------
hs = rd(O / "s01_heavy_summary.csv").set_index(["dataset", "method"])
hp = rd(O / "s01_heavy_paired.csv")
hp = hp[hp.subset == "all_trials"].set_index(["dataset", "baseline"])
labels = [("arbiter_active", "Arbiter"), ("full_cyclic", "Full cyclic"), ("osqp_qp", r"OSQP, $10^{-12}$"),
          ("osqp_qp_v4_stop", r"OSQP, $10^{-9}$"), ("reject_only", "Check only")]
rows = []
for method, label in labels:
    cells = []
    for d in DS:
        r = hs.loc[(d, method)]
        cells.append(f"{ms(r.runtime_p50_ms)} / {ms(r.runtime_p95_ms)}")
    rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
# Verification rates go to the caption; every repairing method except OSQP at 1e-9 must verify every trial.
for method in ["arbiter_active", "full_cyclic", "osqp_qp"]:
    for d in DS:
        assert float(hs.loc[(d, method)].verified_rate) == 1.0, (method, d)
for d in DS:
    assert float(hs.loc[(d, "reject_only")].verified_rate) == 0.0
v9 = [f"{100 * float(hs.loc[(d, 'osqp_qp_v4_stop')].verified_rate):.0f}\\%" for d in DS]
write("tab_heavy", r"""
\begin{table}[t]
\centering
\caption{Repair latency on the 45 model proposals at tolerance $10^{-12}$, median / p95 in ms (15 proposals per dataset, 3 timed repetitions after one warm-up, counterbalanced order). OSQP rows give $\epsilon_{\mathrm{abs}}$. All outputs of Arbiter, full cyclic repair, and OSQP at $10^{-12}$ passed verification; OSQP at $10^{-9}$ passed in """ + f"{v9[0]}, {v9[1]}, and {v9[2]}" + r""" of trials; check only refuses all.}
\label{tab:heavy}
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{@{}lrrr@{}}
\toprule
Method & NYC Taxi & TPC-H SF1 & UCI Adult \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
hr = rd(O / "s01_heavy_raw.csv")
oq = hr[hr.method == "osqp_qp"]
fact("heavy_osqp_iter_cap", {"hit": int((oq.sweeps >= 20000).sum()), "n": int(len(oq)),
                              "by_ds": {SHORT[d]: int((oq[oq.dataset == d].sweeps >= 20000).sum()) for d in DS}}, "nb07 s01_heavy_raw.csv")
ov4 = hr[hr.method == "osqp_qp_v4_stop"]
fact("heavy_osqp_v4stop_verified", [int(b(ov4.verified).sum()), int(len(ov4))], "nb07 s01_heavy_raw.csv")
fact("heavy_p50", {SHORT[d]: {m: float(hs.loc[(d, m)].runtime_p50_ms) for m, _ in labels} for d in DS}, "nb07 s01_heavy_summary.csv")
fact("heavy_speedups", {SHORT[d]: {bb: [float(hp.loc[(d, bb)].speedup_median), float(hp.loc[(d, bb)].speedup_ci95_low), float(hp.loc[(d, bb)].speedup_ci95_high)]
                                   for bb in ["full_cyclic", "osqp_qp", "osqp_qp_v4_stop"]} for d in DS}, "nb07 s01_heavy_paired.csv")
# Price change and worst-case cost.
# l1_adjustment is sum |repaired - proposal| (arbiter_core.py); it is identical across timing repetitions.
_k = ["dataset", "model", "seed", "proposal_index"]
_spread = hr.groupby(_k + ["method"]).l1_adjustment.agg(lambda s: float(s.max() - s.min()))
assert float(_spread.abs().max()) == 0.0, "l1_adjustment must not vary across repetitions"
_l1 = hr.groupby(_k + ["method"]).l1_adjustment.median().unstack("method")
_r_osqp = _l1.arbiter_active / _l1.osqp_qp
_r_cyc = _l1.arbiter_active / _l1.full_cyclic
assert len(_r_osqp) == 45
fact("heavy_l1_ratio", {"vs_osqp": [float(_r_osqp.median()), float(_r_osqp.min()), float(_r_osqp.max())],
                        "vs_full_cyclic": [float(_r_cyc.median()), float(_r_cyc.min()), float(_r_cyc.max())], "proposals": int(len(_r_osqp))},
     "nb07 s01_heavy_raw.csv (L1 norm of the price change, per proposal; median, min, max of the ratio)")
fact("heavy_osqp_runtime_max_ms", float(oq.runtime_ms.max()), "nb07 s01_heavy_raw.csv (OSQP at eps_abs 1e-12, the fallback configuration)")
_dr = rd(O / "s06_dense_raw.csv")
_dfb = _dr[b(_dr.fallback_used) & (_dr.tolerance == 1e-12)]
fact("dense_fallback_worst_ms", {"fallback_max": float(_dfb.fallback_ms.max()), "gate_max": float(_dfb.runtime_ms.max()), "rows": int(len(_dfb))},
     "nb07 s06_dense_raw.csv (rows that used fallback at tolerance 1e-12)")

# --------------------------------------------------------------------------
# Table: scale benchmark at 128 and 4,096 assets (s04)
# --------------------------------------------------------------------------
ss = rd(O / "s04_scale_summary.csv").set_index(["dataset", "scale", "method"])
pp = rd(O / "s04_scale_paired.csv")
pp = pp[pp.subset == "all_trials"].set_index(["dataset", "scale", "baseline"])
bs = build.set_index(["dataset", "scale"])
rows = []
for d in DS:
    for scale in [128, 4096]:
        a = ss.loc[(d, scale, "arbiter_active")]
        fc = ss.loc[(d, scale, "full_cyclic")]
        oqp = ss.loc[(d, scale, "osqp_qp")]
        o9 = ss.loc[(d, scale, "osqp_qp_v4_stop")]
        ro = ss.loc[(d, scale, "reject_only")]
        s1 = pp.loc[(d, scale, "full_cyclic")]
        s2 = pp.loc[(d, scale, "osqp_qp")]
        name = SHORT[d] if scale == 128 else ""
        rows.append(
            f"{name} & {gi(scale)} & {gi(bs.loc[(d, scale), 'constraints'])} & {ms(ro.runtime_p50_ms)} & "
            f"{ms(a.runtime_p50_ms)} / {ms(a.runtime_p95_ms)} & {ms(fc.runtime_p50_ms)} / {ms(fc.runtime_p95_ms)} & "
            f"{ms(oqp.runtime_p50_ms)} / {ms(oqp.runtime_p95_ms)} & {ms(o9.runtime_p50_ms)} ({100 * o9.verified_rate:.0f}) & "
            f"{sp(s1.speedup_median)} [{sp(s1.speedup_ci95_low)}, {sp(s1.speedup_ci95_high)}] & "
            f"{sp(s2.speedup_median)} [{sp(s2.speedup_ci95_low)}, {sp(s2.speedup_ci95_high)}] \\\\"
        )
# tab_scale removed from the paper for length (layout pass); its numbers are cited in the prose from facts.json.

fact("scale", {SHORT[d]: {str(s): {m: [float(ss.loc[(d, s, m)].runtime_p50_ms), float(ss.loc[(d, s, m)].runtime_p95_ms), float(ss.loc[(d, s, m)].verified_rate)]
                                   for m in ["arbiter_active", "full_cyclic", "osqp_qp", "osqp_qp_v4_stop", "reject_only"]}
                          for s in [128, 512, 1024, 2048, 4096]} for d in DS}, "nb07 s04_scale_summary.csv")
fact("scale_speedup", {SHORT[d]: {str(s): {bb: [float(pp.loc[(d, s, bb)].speedup_median), float(pp.loc[(d, s, bb)].speedup_ci95_low), float(pp.loc[(d, s, bb)].speedup_ci95_high)]
                                           for bb in ["full_cyclic", "osqp_qp", "osqp_qp_v4_stop"]} for s in [128, 4096]} for d in DS}, "nb07 s04_scale_paired.csv")
sr = rd(O / "s04_scale_raw.csv")
fact("scale_sweeps_max", sr.groupby("method").sweeps.max().astype(int).to_dict(), "nb07 s04_scale_raw.csv")
fact("scale_v4stop_verified_range", [float(ss.xs("osqp_qp_v4_stop", level=2).verified_rate.min()), float(ss.xs("osqp_qp_v4_stop", level=2).verified_rate.max())], "nb07 s04_scale_summary.csv")
assert int(sr.violations_before.min()) == 1 and int(sr.violations_before.max()) == 51, "fig_scale caption states 1 to 51 violated rows"
fact("scale_viol_before", [int(sr.violations_before.min()), int(sr.violations_before.max()), float(np.median(sr.violations_before))], "nb07 s04_scale_raw.csv")
fact("build_4096_p50", {SHORT[d]: float(bs.loc[(d, 4096), "build_p50_ms"]) for d in DS}, "nb07 s04_build_summary.csv")
_cb = rd(V4 / "03_scalability/constraint_build_trials.csv")
_cb = _cb[_cb.scale == 4096]
fact("snapshot_bytes_4096", {"sparse_numeric_bytes": [int(_cb.sparse_numeric_bytes.min()), int(_cb.sparse_numeric_bytes.max())],
                             "snapshot_estimated_bytes": [int(_cb.snapshot_estimated_bytes.min()), int(_cb.snapshot_estimated_bytes.max())]},
     "v4 03_scalability/constraint_build_trials.csv (4,096-asset snapshots; numeric CSR arrays and total with row metadata)")

# --------------------------------------------------------------------------
# Figure: latency versus scale (s04), one panel per dataset
# --------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# All figure text at 9 pt, the acmart sigconf caption size; the figure is placed at \textwidth (7.0 in).
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 9, "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans", "axes.labelsize": 9,
                            "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9, "axes.linewidth": 0.6})
OKABE = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
         "vermillion": "#D55E00", "purple": "#CC79A7", "black": "#000000"}
series = [("arbiter_active", "Arbiter", OKABE["blue"], "o", "-"),
          ("full_cyclic", "Full cyclic", OKABE["vermillion"], "s", "--"),
          ("osqp_qp", r"OSQP ($10^{-12}$)", OKABE["green"], "^", "-."),
          ("osqp_qp_v4_stop", r"OSQP ($10^{-9}$)", OKABE["orange"], "v", ":"),
          ("reject_only", "Check only", OKABE["black"], "D", "-")]
scales = [128, 512, 1024, 2048, 4096]
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.05), sharey=True, constrained_layout=True)
for ax, d in zip(axes, DS):
    for m, lab, col, mk, ls in series:
        p50 = [ss.loc[(d, s, m)].runtime_p50_ms for s in scales]
        p95 = [ss.loc[(d, s, m)].runtime_p95_ms for s in scales]
        ax.plot(scales, p50, ls, color=col, marker=mk, markersize=3.2, linewidth=1.0, label=lab)
        ax.fill_between(scales, p50, p95, color=col, alpha=0.12, linewidth=0)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(scales)
    ax.set_xticklabels([str(s) for s in scales])
    ax.set_xlabel("Active assets")
    ax.text(0.03, 0.95, SHORT[d], transform=ax.transAxes, va="top", ha="left", fontsize=9)
    ax.grid(True, which="major", linewidth=0.3, alpha=0.5)
axes[0].set_ylabel("Gate latency (ms)")
handles, labs = axes[0].get_legend_handles_labels()
fig.legend(handles, labs, loc="outside upper center", ncol=5, frameon=False, handlelength=2.0, columnspacing=1.0)
fig.savefig(A / "fig_scale.pdf")
plt.close(fig)
(F / "fig_scale.tex").write_text(r"""
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{fig_scale.pdf}
\caption{Gate latency versus snapshot size at $\tol = 10^{-12}$ on attack candidates that violate 1 to 51 rows (median over 120 trials per point; bands reach the p95). OSQP labels give $\epsilon_{\mathrm{abs}}$; both axes are logarithmic.}
\label{fig:scale}
\Description{Three log-log panels, one per dataset, plot median gate latency against 128 to 4,096 assets: Arbiter stays below 1 ms and check only below 0.5 ms, while full cyclic repair and OSQP at two absolute tolerances are 5 to 10 times slower than Arbiter at 128 assets and 70 to 180 times slower at 4,096 assets.}
\end{figure*}
""".strip() + "\n", encoding="utf-8")

# --------------------------------------------------------------------------
# Table: active-row budget on dense candidates (s06)
# --------------------------------------------------------------------------
dov = rd(O / "s06_dense_budget_overview.csv")
fs = dov[dov.variant == "fail_safe"]
cat = rd(O / "s06_dense_catalog.csv")
dense_budget = {}
for budget in [1, 4, 8, 16, 32, 64]:
    for t, msw in [(1e-12, 80), (1e-12, 400), (1e-7, 80)]:
        r = fs[(fs.tolerance == t) & (fs.max_sweeps == msw) & (fs.active_row_budget == budget)].iloc[0]
        cases = int(r.cases)
        dense_budget[f"{t:g}|{msw}|{budget}"] = {"cases": cases, "unverified_by_active_path": int(round(r.fallback_rate * cases)),
                                                 "binding": int(round(r.binding_rate * cases))}
# tab_budget removed from the paper for length (layout pass); its numbers are cited in the prose from facts.json.
fact("dense_budget", dense_budget, "nb07 s06_dense_budget_overview.csv (fail-safe variant)")
fact("dense_catalog", [int(cat.violations_before.min()), int(cat.violations_before.max()), float(np.median(cat.violations_before)),
                       int(cat.drop_duplicates(["dataset", "relation_family", "case_id"]).shape[0])], "nb07 s06_dense_catalog.csv")
fact("dense_fail_safe_verified", float(fs.verified_rate.min()), "nb07 s06_dense_budget_overview.csv")
stages = rd(O / "s06_dense_fallback_stages.csv")
fact("dense_fallback_stages", stages.to_dict(orient="records"), "nb07 s06_dense_fallback_stages.csv")
fp = rd(O / "s07_fallback_paired.csv")
fsum = rd(O / "s07_fallback_summary.csv")
v80 = fp[(fp.fallback_sweeps == 80)]
fact("kernel_speedup_80", [float(v80.speedup_legacy_over_optimized_p50.min()), float(v80.speedup_legacy_over_optimized_p50.max())], "nb07 s07_fallback_paired.csv")
fact("kernel_speedup_80_ci", [float(v80.speedup_ci95_low.min()), float(v80.speedup_ci95_high.max())], "nb07 s07_fallback_paired.csv")
v20 = fsum[fsum.fallback_sweeps == 20]
fact("kernel_verified_20", {m: [int((g.verified_rate == 1.0).sum()), int(len(g))] for m, g in v20.groupby("method")}, "nb07 s07_fallback_summary.csv")
v80s = fsum[fsum.fallback_sweeps == 80]
fact("kernel_verified_80", {m: [int((g.verified_rate == 1.0).sum()), int(len(g))] for m, g in v80s.groupby("method")}, "nb07 s07_fallback_summary.csv")
fact("kernel_p50_80", {m: [float(g.runtime_p50_ms.min()), float(g.runtime_p50_ms.max())] for m, g in v80s.groupby("method")}, "nb07 s07_fallback_summary.csv")
curve = rd(O / "final_sweep_budget_curve.csv")
fact("sweep_curve", curve.to_dict(orient="records"), "nb07 final_sweep_budget_curve.csv")
ab = rd(O / "s05_ablation_summary.csv")
fact("ablation", {f"{SHORT[r.dataset]}|{r.ablation}": [float(r.runtime_p50_ms), float(r.runtime_p95_ms), float(r.verified_rate), int(r.sweeps_max), int(r.trials)]
                  for r in ab.itertuples()}, "nb07 s05_ablation_summary.csv")

# --------------------------------------------------------------------------
# Table: seller price shapes and recompute versus verify (s09, s10)
# --------------------------------------------------------------------------
sp_ = rd(O / "s09_shape_policy.csv")
sgs = rd(O / "s09_shape_gate_summary.csv")
rc = rd(O / "s10_recompute_summary.csv")
rows = []
for shape, label in [("linear", "Linear"), ("concave_sqrt", "Concave"), ("tiered_8", "Tiered")]:
    p = sp_[sp_["shape"] == shape]
    g = sgs[sgs["shape"] == shape]
    r = rc[rc["shape"] == shape]
    viol = p.violated_rows.astype(int)
    viol_txt = "0" if viol.max() == 0 else f"{gi(viol.min())}--{gi(viol.max())}"
    rows.append(
        f"{label} & {viol_txt} & {ms(r.recompute_p50_ms.min())}--{ms(r.recompute_p50_ms.max())} & "
        f"{ms(r.verify_p50_ms.min())}--{ms(r.verify_p50_ms.max())} & {int(round((g.released_rate * g.proposals).sum()))}/{int(g.proposals.sum())} & "
        f"{int(g.sweeps_max.max())} \\\\"
    )
# tab_shapes removed from the paper for length (layout pass); its numbers are cited in the prose from facts.json.
fact("recompute_verify_p50_ms", {sh: {"recompute": [float(rc[rc["shape"] == sh].recompute_p50_ms.min()), float(rc[rc["shape"] == sh].recompute_p50_ms.max())],
                                      "verify": [float(rc[rc["shape"] == sh].verify_p50_ms.min()), float(rc[rc["shape"] == sh].verify_p50_ms.max())]}
                                 for sh in ["linear", "concave_sqrt", "tiered_8"]}, "nb07 s10_recompute_summary.csv")
fact("shape_gate", {sh: {"released": int(round((sgs[sgs["shape"] == sh].released_rate * sgs[sgs["shape"] == sh].proposals).sum())),
                         "proposals": int(sgs[sgs["shape"] == sh].proposals.sum()), "sweeps_max": int(sgs[sgs["shape"] == sh].sweeps_max.max())}
                    for sh in ["linear", "concave_sqrt", "tiered_8"]}, "nb07 s09_shape_gate_summary.csv")

fact("shapes_policy", sp_[["dataset", "shape", "feasible_at_1e12", "violated_rows", "max_residual", "realized_tiers"]].to_dict(orient="records"), "nb07 s09_shape_policy.csv")
mm = rd(O / "s09_shape_model_metrics.csv")
fact("shape_models_r2", {f"{s}|{mo}": [float(g.pooled_r2.min()), float(g.pooled_r2.max())] for (s, mo), g in mm.groupby(["shape", "model"])}, "nb07 s09_shape_model_metrics.csv")
fact("shape_attacks", [int(b(sa.released).sum()), int(len(sa))], "nb07 s09_shape_attacks.csv")

# --------------------------------------------------------------------------
# Table: buyer-history residual snapshots (s11)
# --------------------------------------------------------------------------
rows = []
assert int(hi.released.sum()) == len(hi) and int(hi.fallback_used.sum()) == 0, "tab_history caption states all released without fallback"
for size in [16, 64, 256]:
    g = hi[hi.history_size == size]
    res = g[g.variant == "residual"]
    lc = g[g.variant == "list_cap"]
    assert len(g) == 120
    rows.append(
        f"{size} & {gi(g.derivable_answers.min())}--{gi(g.derivable_answers.max())} & "
        f"{gi(g.remaining_answers.min())}--{gi(g.remaining_answers.max())} & "
        f"{gi(g.residual_rows.min())}--{gi(g.residual_rows.max())} & "
        f"{int(res.sweeps.max())}/{int(lc.sweeps.max())} & {ms(float(np.median(res.gate_ms)))} \\\\"
    )
diag = rd(O / "s11_history_pinned_diagnostic.csv")
write("tab_history", r"""
\begin{table}[t]
\centering
\caption{Buyer-history snapshots at tolerance $10^{-12}$, 120 per row (10 buyers per dataset, proposal, and variant), all released without fallback. Derivable: answers computable from owned ones. Remaining: answers still priced. Rewritten: decompositions rewritten over missing parts. Sweeps: maximum for the residual / list-cap variants. p50: median gate time of the residual variant in ms.}
\label{tab:history}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}rrrrrr@{}}
\toprule
Owned & Derivable & Remaining & Rewritten & Sweeps & p50 \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
fact("history_pinned", [int(b(diag.active_repair_verified).sum()), int(len(diag)), float(diag.max_residual_after.min()), float(diag.max_residual_after.max())], "nb07 s11_history_pinned_diagnostic.csv")

# --------------------------------------------------------------------------
# Table: declaration coverage (s08) and omitted relations (s03)
# --------------------------------------------------------------------------
cov = rd(O / "s08_coverage_raw.csv")
for c in ["released", "fallback_used", "unsafe_release"]:
    cov[c] = b(cov[c])
cov["group"] = cov.encoding.str.replace(r":seed\d+$", "", regex=True)
order = [("families:none", "None"), ("families:D", "D"), ("families:F", "F"), ("families:P", "P"), ("families:DF", "D+F"),
         ("families:DP", "D+P"), ("families:PF", "P+F"), ("fraction:0.25", r"Random 25\%"), ("fraction:0.50", r"Random 50\%"),
         ("fraction:0.75", r"Random 75\%"), ("families:DPF", "All")]
rows = []
for key, label in order:
    g = cov[cov.group == key]
    share_lo, share_hi = g.encoded_share.min(), g.encoded_share.max()
    share = f"{100 * share_lo:.0f}" if round(100 * share_lo) == round(100 * share_hi) else f"{100 * share_lo:.0f}--{100 * share_hi:.0f}"
    n = len(g)
    viol = int((g.complete_violations_after > 0).sum())
    med = float(np.median(g.complete_violations_after))
    cons = int((g.n_conservative_gt_1pct > 0).sum())
    rows.append(f"{label} & {share} & {viol}/{n} & {gi(med)} & {cons}/{n} \\\\")
write("tab_coverage", r"""
\begin{table}[t]
\centering
\caption{Releases gated by a partial declaration, then checked against the complete snapshot at $10^{-12}$ (D, P, F: decomposition, projection, and filter families, alone or combined; random rows: three seeds). Share: declared share of all relations; Exposed: violating releases; Median: violated relations per release; ${>}1\%$: releases with a conservative saving above 1\%.}
\label{tab:coverage}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lrrrr@{}}
\toprule
Declared & Share (\%) & Exposed & Median & ${>}1\%$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
partial = cov[cov.group != "families:DPF"]
fact("coverage_partial", {"encodings": int(partial.groupby(["dataset", "encoding"]).ngroups),
                          "exposed_encodings": int((partial.groupby(["dataset", "encoding"]).complete_violations_after.max() > 0).sum()),
                          "share_max": float(partial.encoded_share.max()),
                          "releases": int(len(partial)), "released": int(partial.released.sum()),
                          "exposed_releases": int((partial.complete_violations_after > 0).sum())}, "nb07 s08_coverage_raw.csv")
fact("coverage_complete", {"releases": int(len(cov[cov.group == "families:DPF"])),
                           "max_violations": int(cov[cov.group == "families:DPF"].complete_violations_after.max())}, "nb07 s08_coverage_raw.csv")
gap = rd(O / "s03_gap_cases.csv")
gap["gap_survives"] = b(gap["gap_survives"])
gap["released_by_incomplete_gate"] = b(gap["released_by_incomplete_gate"])
ogap = rd(V4 / "01_proposals_attacks/omitted_relation_gap_cases.csv")
fact("gap", {"cases": int(len(gap)), "released": int(gap.released_by_incomplete_gate.sum()), "survived": int(gap.gap_survives.sum()),
             "by_type": gap.groupby("relation_type").gap_survives.sum().astype(int).to_dict(),
             "by_type_v4": b(ogap.gap_survives).groupby(ogap.relation_type).sum().astype(int).to_dict(),
             "filter_by_ds": {SHORT[d]: int(gap[(gap.dataset == d) & (gap.relation_type == "filter_derivation")].gap_survives.sum()) for d in DS}},
     "nb07 s03_gap_cases.csv; v4 omitted_relation_gap_cases.csv")

# --------------------------------------------------------------------------
# Table: protocol campaigns (s12 at 1e-12) and protocol overhead (audited)
# --------------------------------------------------------------------------
ps = rd(O / "s12_protocol_summary.csv").set_index("campaign")
e2e = rd(V4 / "02_release_protocol/end_to_end_release.csv")
normal = e2e[e2e.scenario == "normal"]
# Same grouping as the audited paper table: one median per dataset and query profile.
ctl = normal.groupby(["dataset", "query_profile"]).gate_and_commit_ms.median()
ov = rd(CL / "performance/protocol_overhead_summary.csv").set_index("dataset")
ssum = rd(HD / "performance/single_stage_protocol_summary.csv").set_index("size_class")
fault_cases = rd(V4 / "02_release_protocol/fault_injection.csv").case.nunique()
closure_cases = rd(CL / "protocol/request_binding_faults.csv").case.nunique()
rows = [
    f"Token and commit faults & {fault_cases} per dataset & {int(ps.loc['v4_faults', 'passed'])}/{int(ps.loc['v4_faults', 'rows'])} \\\\",
    f"Stale-epoch races & 2 schedules $\\times$ 100 & {int(ps.loc['v4_concurrency', 'passed'])}/{int(ps.loc['v4_concurrency', 'rows'])} \\\\",
    f"Request-binding faults & {closure_cases} per dataset & {int(ps.loc['closure_faults', 'passed'])}/{int(ps.loc['closure_faults', 'rows'])} \\\\",
    f"Cross-service races & 2 schedules $\\times$ 50 & {int(ps.loc['closure_concurrency', 'passed'])}/{int(ps.loc['closure_concurrency', 'rows'])} \\\\",
    f"Delivery authorization & 2 per dataset & {int(ps.loc['closure_delivery', 'passed'])}/{int(ps.loc['closure_delivery', 'rows'])} \\\\",
]
lat = [
    f"Gate and commit, in memory & {ms(ctl.min())}--{ms(ctl.max())} \\\\",
    f"Request-bound durable commit & {ms(ov.complete_p50_ms.min())}--{ms(ov.complete_p50_ms.max())} \\\\",
    f"Single-stage path, scalar to medium & {ms(ssum.loc[['scalar', 'small', 'medium'], 'control_to_commit_p50_ms'].min())}--{ms(ssum.loc[['scalar', 'small', 'medium'], 'control_to_commit_p50_ms'].max())} \\\\",
    f"Single-stage path, {ssum.loc['large', 'payload_bytes'] / 1e6:.2f} MB answer & {ms(ssum.loc['large', 'control_to_commit_p50_ms'])} \\\\",
]
write("tab_protocol", r"""
\begin{table}[t]
\centering
\caption{Protocol behavior. Top: campaigns rerun at $10^{-12}$; a case passes when the gate releases exactly when the specification requires. Bottom: median control-path latency in ms from the audited runs on 2-vCPU hosts at $10^{-7}$, excluding query execution.}
\label{tab:protocol}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lrr@{}}
\toprule
Campaign & Cases & Passed \\
\midrule
""" + "\n".join(rows) + r"""
\midrule
\multicolumn{2}{@{}l}{Control path} & p50 (ms) \\
\midrule
""" + "\n".join(f"\\multicolumn{{2}}{{@{{}}l}}{{{r.split(' & ')[0]}}} & {r.split(' & ')[1]}" for r in lat) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
fact("protocol", {k: [int(ps.loc[k, "passed"]), int(ps.loc[k, "rows"])] for k in ps.index}, "nb07 s12_protocol_summary.csv")
fact("protocol_latency", {"gate_commit_p50": [float(ctl.min()), float(ctl.max())],
                          "durable_complete_p50": [float(ov.complete_p50_ms.min()), float(ov.complete_p50_ms.max())],
                          "durable_complete_p95": [float(ov.complete_p95_ms.min()), float(ov.complete_p95_ms.max())],
                          "single_stage_p50": {k: float(ssum.loc[k, "control_to_commit_p50_ms"]) for k in ssum.index},
                          "single_stage_bytes": {k: int(ssum.loc[k, "payload_bytes"]) for k in ssum.index}},
     "v4 end_to_end_release.csv; closure protocol_overhead_summary.csv; hardening single_stage_protocol_summary.csv")
conc = rd(V4 / "02_release_protocol/stale_epoch_concurrency.csv")
fact("races_released", conc.groupby("schedule").released.sum().astype(int).to_dict(), "v4 stale_epoch_concurrency.csv")
vec = rd(O / "s12_protocol_vectors.csv")
fact("protocol_vectors", vec.to_dict(orient="records"), "nb07 s12_protocol_vectors.csv")
inc = rd(V4 / "03_scalability/incremental_update_trials.csv")
fact("incremental", {"speedup_median": float(np.median(inc.speedup)), "speedup_by_batch": inc.groupby("batch_relations").speedup.median().to_dict(),
                     "identical": [int(b(inc.same_constraint_row_multiset).sum()), int(len(inc))]}, "v4 incremental_update_trials.csv (2 vCPU)")

# --------------------------------------------------------------------------
# Model metrics (audited proposals)
# --------------------------------------------------------------------------
mp = rd(V4 / "01_proposals_attacks/model_metrics_pooled.csv")
fact("model_r2", {f"{SHORT[d]}|{mo}": [float(g.pooled_r2.min()), float(g.pooled_r2.max())] for (d, mo), g in mp.groupby(["dataset", "model"])}, "v4 model_metrics_pooled.csv")
fact("model_negative_max", int(mp.negative_predictions.max()), "v4 model_metrics_pooled.csv")

env = json.loads((RUN / "environment.json").read_text())
fact("nb07_env", {"cpu": env["cpu_model"], "logical_cpus": env["logical_cpu_count"], "memory_gb": env["memory_bytes"] / 1e9,
                  "python": env["python"], "packages": env["packages"], "stage_seconds": sum(env["stage_wall_seconds"].values())}, "nb07 environment.json")
(PAPER / "scripts").mkdir(exist_ok=True)
(PAPER / "scripts" / "facts.json").write_text(json.dumps(facts, indent=1, default=str), encoding="utf-8")
print("tables:", sorted(p.name for p in T.glob("*.tex")))
print("facts:", len(facts))
