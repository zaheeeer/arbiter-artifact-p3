# Extended version

`extended_version.pdf` accompanies the PVLDB submission. It holds the full proofs, the model hyperparameters, the base-workload results that the paper summarizes, the extension campaign in full, and the full comparison of pricing approaches. Lemma, theorem, section, and equation numbers refer to the paper.

## Regenerating its tables and building it

`source/` contains the LaTeX source and the scripts that write every table, the scale figure, and the two facts files from the evidence packages. From the repository root:

```bash
cd extended_version/source
python3 scripts/make_floats.py ../../evidence/07_revision_full_colab-cpu-primary ../../evidence/arbiter_icde_evidence_full_audited_v4 ../../evidence/05_request_bound_closure_full_colab-cpu-primary ../../evidence/06_performance_hardening_full_colab-cpu-primary .
python3 scripts/make_step2.py ../../evidence/08_step2_full_colab-cpu-primary .
python3 scripts/make_extended.py .
cd extended
pdflatex extended_version
pdflatex extended_version
```

The scripts need NumPy, pandas, and matplotlib, and the build needs a standard TeX installation.

- `make_floats.py` writes every data table of the paper to `floats/tables/`, the paper's scale figure (Figure 2) to `floats/figures/` and `assets/`, and `scripts/facts.json`. The extended version uses three of these tables (buyer histories, revision-campaign proposals, and protocol campaigns) and refers to the figure as Figure 2 of the paper.
- `make_step2.py` writes the five extension-campaign tables to `extended/` and `scripts/facts_step2.json`.
- `make_extended.py` writes the active-row budget table to `extended/` from `facts.json`.

`facts.json` and `facts_step2.json` list every number that the paper's prose cites, each with the evidence file that produced it. Rerunning the scripts reproduces the committed `.tex` and `.json` files byte for byte. `fig_scale.pdf` differs only in its embedded creation date.
