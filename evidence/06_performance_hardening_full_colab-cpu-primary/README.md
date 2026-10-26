# ARBITER performance hardening full run

This targeted Notebook 06 campaign preserves audited-v4 Notebooks 00-04 and
request-bound closure Notebook 05 v1.1 unchanged. The signed token protocol
remains byte-identical to v1.1. The primary performance path calls
issue_token once and instruments its single internal durable stage; it never
performs the duplicate staging transaction present in the earlier benchmark.

Four preregistered supported assets produce scalar, small, medium, and large
answers by actual execution over the audited DuckDB database. Query execution
occurs once per profile and is recorded separately; it is not included in the
protocol control-path latency.

Fallback timings are paired kernel-only measurements from identical active
intermediates. The optimized path reuses the audited active cyclic kernel with
a budget covering all current violations, then independently verifies, invokes
OSQP if necessary, and otherwise rejects.

Observed P95 uses an empirical, non-interpolated order statistic (NumPy
method='higher'). Median 95% intervals use deterministic percentile bootstrap.
All measured repetitions are retained; only declared warmups are excluded.

Claims remain limited to this local SQLite outbox prototype, four answer
profiles, four exercised fallback conditions, and the recorded environment.
Network delivery, crash recovery, payment settlement, distributed atomicity,
general SQL, standard TPC-H queries, and cross-machine generalization remain
unsupported.

Environment label: colab_cpu_primary
Hardening mode: full
Notebook 04 source: nested_notebook04_package
Notebook 05 source: direct_notebook05_output
