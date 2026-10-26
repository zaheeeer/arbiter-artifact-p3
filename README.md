# Arbiter: Release-Time Enforcement of Query Prices in Data Markets

Artifact repository of the PVLDB submission

> Zaheer Ahmad Gondal, Xiaofeng Ding, Attiq Ur Rehman, and Jamil Ahmad Memon.
> *Arbiter: Release-Time Enforcement of Query Prices in Data Markets.*
> School of Computer Science and Technology, Huazhong University of Science and Technology, Wuhan, China.

Arbiter is a release-time control plane for query-based data markets. It compiles queries of a single-table SQL fragment, which includes one-column `GROUP BY`, into certified pricing relations. Its release gate repairs only the violated rows of a supplied price vector. An answer leaves the system only after an independent verifier and a request-bound, epoch-bound token admit the price.

This repository holds the code, the notebooks that ran every experiment, the raw results, scripts that check those results, and the extended version of the paper.

## Contents

| Path | Contents |
|---|---|
| `extended_version/` | Extended version of the paper (PDF and LaTeX source): full proofs, model hyperparameters, base-workload results that the paper summarizes, and the extension campaign in full |
| `src/` | The Python modules of Arbiter and of the experiment campaigns, one copy of each |
| `notebooks/` | The Google Colab notebooks that ran the campaigns, numbered in run order (00 to 08) |
| `evidence/` | The five evidence packages the notebooks wrote: generated SQL assets, compiled relations, proposals, every raw trial table, recorded environments, and SHA-256 manifests |
| `reproduction/` | Scripts that check the evidence without rerunning the campaigns, with their recorded output |
| `SHA256SUMS.txt` | SHA-256 of every file in this repository |

## Campaigns

The paper reports three campaigns. Each campaign ran in Colab sessions on its own hosts, and the paper never compares timings across campaigns.

| Campaign in the paper | Notebooks | Evidence package | Host |
|---|---|---|---|
| Audited campaign | 00 to 04 | `evidence/arbiter_icde_evidence_full_audited_v4/` | Colab CPU runtime, 2 vCPUs, Python 3.12.13 |
| Audited campaign: request-bound release | 05 | `evidence/05_request_bound_closure_full_colab-cpu-primary/` | Intel Xeon at 2.20 GHz, 2 vCPUs, Python 3.12.13 |
| Audited campaign: performance hardening | 06 | `evidence/06_performance_hardening_full_colab-cpu-primary/` | AMD EPYC 7B12, 2 vCPUs, Python 3.12.13 |
| Revision campaign | 07 | `evidence/07_revision_full_colab-cpu-primary/` | Intel Xeon at 2.20 GHz, 8 vCPUs, 54.8 GB, Python 3.13.16 |
| Extension campaign | 08 | `evidence/08_step2_full_colab-cpu-primary/` | AMD EPYC 7B12, 8 vCPUs, 54.8 GB, Python 3.13.16 |

- The audited campaign built the workload, validated the compiler, trained the price models, and ran the protocol campaigns.
- The revision campaign re-executed every experiment whose outcome can depend on the verifier tolerance, at tolerance 1e-12. It also measured every repair, verification, and snapshot-build latency that the paper reports on the base workload.
- The extension campaign ran the `GROUP BY` extension, the multi-part chain audit, the price-floor study, the comparison against the exact projection with the interior-point solver Clarabel, and the scale-out to 24,576 assets.

Every package records its environment (`environment.json`), its run configuration (`run_config.json` or `*_run_config.json`), and the result of each integrity check (`stage_quality_checks.csv`). The packages of Notebooks 00 to 07 also contain a `claim_audit.csv`, which states the claims that the evidence supports and those it does not. Package names are the working names under which the campaigns ran. Each package contains exactly the files listed in its `checksums.sha256`, byte for byte, plus the checksum files themselves.

## Where each number comes from

`extended_version/source/scripts/facts.json` and `facts_step2.json` record every number that the paper's prose cites, together with the evidence file that produced it. The table generators in the same folder read only the evidence packages. In brief:

| Paper results | Evidence |
|---|---|
| Table 3 (datasets and compiled workload) | audited `00_workload/`; revision `outputs/s04_build_summary.csv` |
| RQ1: model proposals, tolerance (Table 5), per-row attacks, release outcomes (Table 4) | revision `outputs/s01_*`, `s02_*`, `s09_*`, `s11_*` |
| RQ1: grouped aggregates and price floors | extension `outputs/s21_*`, `s23_*` |
| RQ2: scale and snapshot build | revision `outputs/s04_*`; audited `03_scalability/constraint_build_trials.csv` |
| RQ2: scale-out, proposal scale, and price change | extension `outputs/s25_*`, `s24_*` |
| RQ2: protocol | revision `outputs/s12_*`; audited `02_release_protocol/`; request-bound release `protocol/`, `performance/`; hardening `performance/single_stage_protocol_*` |
| RQ3: relation families, active rows, dense violations, fallback, sweep budget | revision `outputs/s05_*`, `s06_*`, `s07_*`, `s10_*`, `final_sweep_budget_curve.csv` |
| RQ4: compiler validity | audited `00_workload/`; extension `outputs/s21_*` |
| RQ4: partial declarations (Table 6) and omitted relations | revision `outputs/s08_*`, `s03_*` |
| RQ4: multi-part chains and audit cost | extension `outputs/s22_*` |

## Checking the results without rerunning

The checks need Python 3 with NumPy, pandas, and matplotlib (`pip install -r reproduction/requirements.txt`). We ran them with Python 3.13.16, NumPy 2.5.3, pandas 3.0.5, and matplotlib 3.11.2. Run every command from the repository root.

1. **Integrity of the evidence.** Each package lists the SHA-256 of its files:

   ```bash
   (cd evidence/07_revision_full_colab-cpu-primary && sha256sum -c checksums.sha256)
   ```

   On macOS, use `shasum -a 256 -c checksums.sha256`. The same command works in every package folder.

2. **Independent recomputation of the revision campaign.** A script that imports no Arbiter module re-derives all 26 summary tables of Notebook 07 from its raw rows, including every bootstrap interval from its documented seed:

   ```bash
   python3 reproduction/audit_recompute.py evidence/07_revision_full_colab-cpu-primary evidence/arbiter_icde_evidence_full_audited_v4
   ```

   The last line reads `SUMMARY: 26/26 recomputations match`. The full output is in `reproduction/audit_recompute_output.txt`.

3. **Second execution of Notebook 07.** `reproduction/second_execution_nb07/` holds the raw outputs of a second execution of Notebook 07 on a different machine with newer library versions. The comparison script excludes timing columns:

   ```bash
   python3 reproduction/compare_second_execution.py evidence/07_revision_full_colab-cpu-primary/outputs reproduction/second_execution_nb07/outputs
   ```

   Every compared table is identical except the rows that depend on OSQP, which the second execution replaced by a deterministic stand-in, and the rows that depend on retrained models (stage s09 and the two s09 rows of the sweep-budget curve). `reproduction/README.md` explains the details.

4. **Tables of the extended version.** `extended_version/README.md` gives the commands that regenerate every table, the scale figure, and both facts files from the evidence, and then build the PDF.

## Rerunning the campaigns

The notebooks run on Google Colab with a CPU runtime and a Google Drive folder. `notebooks/README.md` gives the order, the inputs of each notebook, and the run modes. In short:

- Every notebook first runs in a short pilot mode and then in full mode. Pilot results test the workflow only.
- Notebook 00 downloads the public data, checks it against fixed SHA-256 values, generates TPC-H SF1 with the DuckDB TPC-H extension, and writes the shared modules to Drive. Notebooks 01 to 04 complete the audited campaign.
- Notebooks 05 to 08 read the audited package from Drive and verify it file by file against its SHA-256 manifest before they start; each package records that check in its input audit file. Notebooks 07 and 08 also confirm that the audited package is byte-identical after the run.
- Notebooks 07 and 08 checkpoint each part to Drive and resume after a disconnection.

## Data

This repository contains no raw data. The notebooks obtain it:

| Dataset | Source | SHA-256 of the file used |
|---|---|---|
| NYC Yellow Taxi trips, January 2024 (2,964,624 rows) | NYC Taxi and Limousine Commission trip records, `yellow_tripdata_2024-01.parquet` | `c4d59da7bbc8abaeeeb1727947ee93d9891a71acb42854bd80db1571b2030510` |
| UCI Adult (48,842 rows) | UCI Machine Learning Repository, `adult.zip` | `7537312dd56c2b98035880805ce99e68183a30ee468aa5329d6df0fbb3cc21bb` |
| TPC-H SF1 (8,661,245 rows in eight tables; the workload reads `lineitem`, `orders`, and `customer`, 7,651,215 rows) | Generated by the DuckDB TPC-H extension (DuckDB 1.3.2) | generated, not downloaded |

The source URLs are recorded in `evidence/arbiter_icde_evidence_full_audited_v4/00_workload/dataset_metadata.csv`.

## Scope of the evidence

- Seller list prices and attack centers are controlled experimental constructions, not market observations, and TPC-H is benchmark-generated.
- The per-row box attack is exact for one row; it is not a worst case over all rows or a model of buyer behavior.
- The compiler results hold for the declared single-table fragment and its construction oracles, not for arbitrary SQL.
- The release protocol is an in-process prototype with a local SQLite log. The evidence covers no network delivery, distributed atomicity, or payment settlement.
- Each campaign's timings come from one host per run, so they do not establish performance across hardware.
