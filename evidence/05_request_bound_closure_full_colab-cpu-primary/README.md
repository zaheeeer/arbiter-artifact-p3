# ARBITER request-bound closure full run

This targeted campaign preserves the audited-v4 00-04 evidence and evaluates
only the closure layer. It uses actual DuckDB query answers over the already
audited NYC Taxi, UCI Adult, and TPC-H SF1 database. It has no synthetic-data or
synthetic-answer fallback. Prices, dense attacks, and tolerance margins are
controlled deterministic experimental constructions, not market observations.

The durable claim ends at a local SQLite transaction that records the purchase
authorization and moves the staged answer into a released outbox. Network
delivery, distributed transactions, and payment settlement are not claimed.

Environment label: colab_cpu_primary
Closure mode: full
Active run: full_audited_v4
Audited evidence package: arbiter_icde_evidence_full_audited_v4
Evidence source: nested_notebook04_package
