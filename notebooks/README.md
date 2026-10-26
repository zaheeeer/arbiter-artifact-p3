# Notebooks

These are the Google Colab notebooks that produced the evidence packages in `../evidence/`. Notebooks 00 to 05 are the executed copies, with their outputs. Notebooks 06 to 08 are the notebook files without outputs; their results are the evidence packages. Colab account metadata (user name and user ID) and widget state were removed from every notebook. The eight DuckDB progress bars in Notebooks 00, 02, and 05 therefore show in their plain-text form. Code, text, and all other outputs are unchanged.

| Notebook | Campaign | Writes |
|---|---|---|
| `00_START_HERE_ARBITER_AUDITED.ipynb` | Audited: data, workload, compiler | `00_workload/` of the audited package |
| `01_Proposal_and_Certified_Attacks.ipynb` | Audited: price models and per-row attacks | `01_proposals_attacks/` |
| `02_Atomic_Release_Protocol_and_Faults.ipynb` | Audited: release protocol, faults, stale epochs | `02_release_protocol/` |
| `03_Scalability_Baselines_and_Ablations.ipynb` | Audited: scale, baselines, ablations | `03_scalability/` |
| `04_Evidence_Audit_and_Package.ipynb` | Audited: claim audit and packaging | `04_evidence/` and the package ZIP |
| `05_Request_Bound_Release_Closure_AUDITED_Fixed_v1_1.ipynb` | Audited: request-bound release | `05_request_bound_closure_full_colab-cpu-primary/` |
| `06_Performance_Hardening_AUDITED_v1_0.ipynb` | Audited: performance hardening | `06_performance_hardening_full_colab-cpu-primary/` |
| `07_Arbiter_Revision_Campaign_v1_0.ipynb` | Revision campaign (stages s01 to s12) | `07_revision_full_colab-cpu-primary/` |
| `08_Arbiter_Step2_Campaign_v1_1.ipynb` | Extension campaign (stages s21 to s25) | `08_step2_full_colab-cpu-primary/` |

Each notebook carries the code it runs. Notebook 00 writes the four audited modules to Drive for Notebooks 01 to 04, and Notebooks 05 to 08 embed the modules they use. Every evidence package stores the modules it ran in `code/`, with their SHA-256 in `code/code_manifest.json`, and these files equal the files in `../src/`.

## Running them

Use a Colab CPU runtime (no GPU) and run each notebook with **Runtime > Run all** after **Runtime > Disconnect and delete runtime**. Every notebook keeps its files under the Drive folder `MyDrive/Arbiter_ICDE_2027/`, the working folder name of all campaigns. To use another folder, set the environment variable `ARBITER_PROJECT_ROOT` before running.

1. **Notebooks 00 to 04 (audited campaign).** Set `RUN_MODE = "pilot"` in Notebook 00 and run 00 to 04 in order; each prints `Notebook NN PASS`. Then set `RUN_MODE = "full"` and run 00 to 04 again. Notebook 00 downloads NYC Taxi (January 2024) and UCI Adult, rejects any file whose SHA-256 differs from the audited value, generates TPC-H SF1 with the DuckDB TPC-H extension, and writes the shared modules to `MyDrive/Arbiter_ICDE_2027/code/`. The full run writes `runs/full_audited_v4/arbiter_icde_evidence_full_audited_v4.zip`.
2. **Notebook 05 (request-bound release).** Needs the full audited run and its DuckDB database. Run with `CLOSURE_MODE = "pilot"`, then `"full"`.
3. **Notebook 06 (performance hardening).** Needs the full audited run, the full output of Notebook 05, the audited DuckDB database, and the two downloaded data files. Run with `HARDENING_MODE = "pilot"`, then `"full"`.
4. **Notebook 07 (revision campaign).** Needs the audited package, either as the folder `runs/full_audited_v4/arbiter_icde_evidence_full_audited_v4/` or as its ZIP in `inputs/`. Run with `RUN_MODE = "pilot"`, then `"full"`; the full run refuses to start without a passing pilot. Finished parts are checkpointed to Drive, so after a disconnection **Run all** resumes. The full run writes `runs/revision_v1/arbiter_revision_full_colab-cpu-primary_v1_0.zip`.
5. **Notebook 08 (extension campaign).** Needs the audited package as for Notebook 07. Stage s21 executes queries on the real data: the notebook copies `yellow_tripdata_2024-01.parquet` and `adult.zip` from `inputs/` or downloads them, checks both against the audited SHA-256 values, and regenerates TPC-H SF1. Run with `RUN_MODE = "pilot"`, then `"full"`. The full run writes `runs/step2_v1/arbiter_step2_full_colab-cpu-primary_v1_1.zip`.

Keep the environment label `colab_cpu_primary` in Notebooks 05 to 08; their package names derive from it. Pilot results test the workflow only. Each campaign was run as one pilot followed by one full run, and the full run is the evidence.

Recorded stage times of the full runs: 3,532 s for Notebook 07 (72 parts) and 3,918 s for Notebook 08 (28 parts), on the hosts listed in the top-level README.
