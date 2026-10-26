"""Write the generated table of the extended version from scripts/facts.json.

Usage: python3 -I make_extended.py PAPER_DIR

The extended version (extended/extended_version.tex) holds the full proofs, the
model hyperparameters, and the two RQ3 ablations that the paper summarizes. Its
only data table is generated here from facts.json, so every number keeps the
provenance recorded by make_floats.py.
"""
import json
import sys
from pathlib import Path

PAPER = Path(sys.argv[1])
facts = json.loads((PAPER / "scripts" / "facts.json").read_text(encoding="utf-8"))
budget = facts["dense_budget"]["value"]
cat = facts["dense_catalog"]["value"]
settings = [("1e-12", "80", r"$10^{-12}$, 80"), ("1e-12", "400", r"$10^{-12}$, 400"), ("1e-07", "80", r"$10^{-7}$, 80")]
rows = []
for b in [1, 4, 8, 16, 32, 64]:
    cells = []
    for tol, sweeps, _ in settings:
        r = budget[f"{tol}|{sweeps}|{b}"]
        cells.append(f"{r['unverified_by_active_path']} / {r['cases']}")
    binding = budget[f"1e-12|80|{b}"]["binding"]
    rows.append(f"{b} & " + " & ".join(cells) + f" & {binding} \\\\")
cases = {budget[k]["cases"] for k in budget}
assert cases == {18}, cases
assert facts["dense_fail_safe_verified"]["value"] == 1.0
text = r"""
\begin{table}[h]
\centering
\caption{Active-row budget under dense violations: candidates left unverified by the active path, out of 18 per setting (""" + f"{cat[0]} to {cat[1]}" + r""" violated rows per candidate). The fail-safe tier verified every candidate that the active path left unverified. The last column counts candidates whose violated set exceeded the budget at $10^{-12}$ with 80 sweeps.}
\label{tab:ext-budget}
\begin{tabular}{@{}rrrrr@{}}
\toprule
Budget $B$ & """ + " & ".join(s[2] for s in settings) + r""" & Budget binding \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
(PAPER / "extended").mkdir(exist_ok=True)
(PAPER / "extended" / "tab_dense_budget.tex").write_text(text.strip() + "\n", encoding="utf-8")
print("wrote extended/tab_dense_budget.tex")
