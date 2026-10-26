# Checking the evidence without rerunning

Run every command from the repository root. The scripts need Python 3 with NumPy and pandas (`pip install -r reproduction/requirements.txt`); we ran them with Python 3.13.16, NumPy 2.5.3, and pandas 3.0.5.

## Independent recomputation of the revision campaign

`audit_recompute.py` imports no Arbiter module. It re-derives all 26 summary tables of Notebook 07 from the raw rows with its own code, reproduces every bootstrap interval from its documented seed, compares stages s01 to s03 with the audited records, and recomputes the values behind the claim audit.

```bash
python3 reproduction/audit_recompute.py evidence/07_revision_full_colab-cpu-primary evidence/arbiter_icde_evidence_full_audited_v4
```

It prints one line per check and ends with `SUMMARY: 26/26 recomputations match`. `audit_recompute_output.txt` holds the recorded output.

## Second execution of Notebook 07

`second_execution_nb07/` holds the raw outputs and run records of a second full execution of Notebook 07 (version 1.0) on a different machine: a Linux virtual machine with 2 vCPUs (Intel Xeon at 2.80 GHz) and 8.4 GB of memory, running Python 3.13.16, NumPy 2.5.3, pandas 3.0.5, SciPy 1.18.1, and scikit-learn 1.9.1. OSQP was not installed on that machine, so the run used a deterministic stand-in for it (`osqp_test_double` in `environment.json`). The run kept the notebook's default environment label, `colab_cpu_primary`, although it did not run on Colab.

`compare_second_execution.py` joins the two executions on the key columns of each table, excludes timing columns and the rows of OSQP baselines, and counts the cells that differ:

```bash
python3 reproduction/compare_second_execution.py evidence/07_revision_full_colab-cpu-primary/outputs reproduction/second_execution_nb07/outputs
```

`second_execution_comparison.txt` holds the recorded output. It matches 27,759 keyed rows. Every compared table is identical except these:

- `s06_dense_raw` (5 rows) and `s06_dense_optimized` (4 rows): fallback cases where real OSQP returned a vector that verified and the stand-in did not.
- `s09_shape_model_metrics` and `s09_shape_gate`: the price models are retrained on each machine, and the retrained proposals differ.
- `final_sweep_budget_curve`: the two rows built from the s09 proposals (concave and tiered shapes).

The second execution's timings are not used anywhere.
