# Source modules

One copy of every Python module that the campaigns ran. Each file is byte-identical to the copy stored in the `code/` folder of the evidence packages that used it; `SHA256` below lists the hashes, which also appear in each package's `code/code_manifest.json`.

| Module | Contents | First used by |
|---|---|---|
| `arbiter_workload.py` | Supported SQL fragment (parsed with SQLGlot), deterministic workload generator, relation compiler (rules R1 to R3), construction oracle, and executed SQL checks | Notebook 00 |
| `arbiter_data.py` | Download and SHA-256 check of NYC Taxi and UCI Adult; DuckDB preparation, including TPC-H SF1 | Notebook 00 |
| `arbiter_core.py` | Constraint snapshot, active-row repair and the baselines (check only, full cyclic repair, OSQP), independent verifier, exact per-row box attacks, release token, answer buffer, and commit | Notebook 00 |
| `arbiter_experiments.py` | Audited experiments: controlled prices, grouped out-of-fold proposals, attack campaigns, scale benchmark, fault injection, and stale-epoch races | Notebook 01 |
| `arbiter_request_release.py` | Request-bound release protocol: request context, bound answer buffer, request-bound token, and durable release ledger | Notebook 05 |
| `arbiter_closure_experiments.py` | Request-binding faults, cross-service races, tiered fail-safe repair, active-row budget ablation, tolerance sensitivity, and protocol overhead | Notebook 05 |
| `arbiter_performance_hardening.py` | Single-stage protocol path, fallback kernels, and optimized fail-safe validation | Notebook 06 |
| `arbiter_revision.py` | Revision campaign: stages s01 to s12 at tolerance 1e-12, checkpointing, and integrity checks | Notebook 07 |
| `arbiter_step2.py` | Extension campaign: `GROUP BY` extension (rules R4 to R6), multi-part chain audit, price floors, exact-projection bracket with Clarabel, and scale-out | Notebook 08 |

The notebooks set up these modules themselves; see `../notebooks/README.md`.

## SHA256

| Module | SHA-256 |
|---|---|
| `arbiter_workload.py` | `cd3a78b28c573698e77fe02bc301273bbc959d770ee6daadae52ace68151ef5a` |
| `arbiter_data.py` | `fb4d85f6a36be12bf1dc3ac09e80383f4d7e52e83a923bae1840a87443452c86` |
| `arbiter_core.py` | `b6797381dd883eac5d3cef7f1f86eaa2ad66bf73c498fdcfb7493f5113175ffa` |
| `arbiter_experiments.py` | `f5919e83b4c2658950a611e94f3bc8de62000a4ab7bd1cb24b90f64e5a0b590e` |
| `arbiter_request_release.py` | `b00dc16f4ab5ea9fc57d30fb141c83f31003bc8014e8285c1e1eabd812f66fc3` |
| `arbiter_closure_experiments.py` | `bf63760ee2124e6976bcf02e17d14c00c083defa4bdedae70fedfdec46e7cfb4` |
| `arbiter_performance_hardening.py` | `e4460d788b473225e913e0289d9e8f4b125be5f2b6b61095d66885026e764983` |
| `arbiter_revision.py` | `8ed288e64963bf44155f99529866adeab2931ca5ca97a6f027d6279e67e26e4d` |
| `arbiter_step2.py` | `3c3152a53b1b61ca3c01084bc9917f9f86087b846f6b78d57af88e0d556e46b2` |
