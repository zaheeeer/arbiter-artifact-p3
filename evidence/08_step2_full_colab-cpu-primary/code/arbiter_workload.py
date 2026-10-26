"""Supported SQL fragment, deterministic workload generator, and relation compiler.

The compiler intentionally supports a small fragment with explicit derivation
rules.  Queries outside the fragment are reported as unsupported rather than
being guessed from footprint similarity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations, cycle, product
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    import sqlglot
    from sqlglot import exp
except Exception:  # pragma: no cover - notebooks install the pinned dependency.
    sqlglot = None
    exp = None


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _sql_literal(value: Any) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            raise ValueError("SQL numeric literals must be finite")
        # Python's repr is a shortest round-trippable decimal, so the SQLGlot
        # compiler reconstructs the exact descriptor boundary used by the oracle.
        return repr(float(value))
    return "'" + str(value).replace("'", "''") + "'"


@dataclass(frozen=True)
class QueryDescriptor:
    table: str
    mode: str
    output_columns: tuple[str, ...]
    aggregate: str
    measure_column: str
    predicate_kind: str
    predicate_column: str
    lower: float | None
    upper: float | None
    categories: tuple[str, ...]

    def to_json(self) -> str:
        value = asdict(self)
        value["output_columns"] = list(self.output_columns)
        value["categories"] = list(self.categories)
        return _canonical_json(value)

    @classmethod
    def from_json(cls, value: str) -> "QueryDescriptor":
        parsed = json.loads(value)
        parsed["output_columns"] = tuple(parsed.get("output_columns", []))
        parsed["categories"] = tuple(parsed.get("categories", []))
        return cls(**parsed)

    def relation_group_key(self) -> tuple[Any, ...]:
        return (
            self.table,
            self.mode,
            self.output_columns,
            self.aggregate,
            self.measure_column,
            self.predicate_column,
            self.predicate_kind,
        )

    def predicate_key(self) -> tuple[Any, ...]:
        return (
            self.predicate_kind,
            self.predicate_column,
            self.lower,
            self.upper,
            self.categories,
        )


def _strip_alias(expression):
    return expression.this if exp is not None and isinstance(expression, exp.Alias) else expression


def _literal_value(node) -> Any:
    if exp is not None and isinstance(node, exp.Neg):
        value = _literal_value(node.this)
        if isinstance(value, str):
            raise ValueError("a string literal cannot be negated")
        return -float(value)
    if exp is None or not isinstance(node, exp.Literal):
        raise ValueError("predicate bound is not a literal")
    if node.is_string:
        return str(node.this)
    return float(node.this)


def _column_name(node) -> str:
    if exp is None or not isinstance(node, exp.Column):
        raise ValueError("expected a column reference")
    return str(node.name)


def _flatten_and(node) -> list[Any]:
    if exp is not None and isinstance(node, exp.And):
        return _flatten_and(node.left) + _flatten_and(node.right)
    return [node]


def _parse_predicate(select) -> dict[str, Any]:
    where = select.args.get("where")
    if where is None:
        return {
            "predicate_kind": "all",
            "predicate_column": "",
            "lower": None,
            "upper": None,
            "categories": (),
        }
    terms = _flatten_and(where.this)
    if len(terms) == 2:
        lower = None
        upper = None
        column = None
        for term in terms:
            if isinstance(term, exp.GTE):
                term_column = _column_name(term.left)
                lower = float(_literal_value(term.right))
            elif isinstance(term, exp.LT):
                term_column = _column_name(term.left)
                upper = float(_literal_value(term.right))
            else:
                raise ValueError("only >= lower AND < upper ranges are supported")
            if column is None:
                column = term_column
            elif column != term_column:
                raise ValueError("range bounds must reference the same column")
        if lower is None or upper is None or not lower < upper:
            raise ValueError("invalid half-open numeric interval")
        if not math.isfinite(lower) or not math.isfinite(upper):
            raise ValueError("numeric interval bounds must be finite")
        return {
            "predicate_kind": "numeric_interval",
            "predicate_column": str(column),
            "lower": lower,
            "upper": upper,
            "categories": (),
        }
    if len(terms) != 1:
        raise ValueError("unsupported predicate conjunction")
    term = terms[0]
    if isinstance(term, exp.EQ):
        column = _column_name(term.left)
        value = str(_literal_value(term.right))
        return {
            "predicate_kind": "category_set",
            "predicate_column": column,
            "lower": None,
            "upper": None,
            "categories": (value,),
        }
    if isinstance(term, exp.In):
        column = _column_name(term.this)
        values = tuple(
            sorted(set(str(_literal_value(item)) for item in term.expressions))
        )
        if not values:
            raise ValueError("IN list must not be empty")
        return {
            "predicate_kind": "category_set",
            "predicate_column": column,
            "lower": None,
            "upper": None,
            "categories": values,
        }
    raise ValueError("unsupported predicate form")


def descriptor_from_sql(sql: str) -> QueryDescriptor:
    """Parse one query from the supported fragment using SQLGlot's DuckDB dialect."""

    if sqlglot is None or exp is None:
        raise RuntimeError("sqlglot is required for SQL compilation")
    tree = sqlglot.parse_one(sql, read="duckdb")
    if not isinstance(tree, exp.Select):
        raise ValueError("only a top-level SELECT is supported")

    # Fail closed on relational structure that is not represented by
    # QueryDescriptor. Counting only base tables is insufficient: a subquery or
    # CTE can contain one table while changing LIMIT, DISTINCT, aggregation, or
    # filtering semantics before the outer SELECT.
    forbidden_nodes = {
        "CTE",
        "Lateral",
        "Pivot",
        "Subquery",
        "TableSample",
        "Unnest",
        "Window",
        "With",
    }
    if any(type(node).__name__ in forbidden_nodes for node in tree.walk()):
        raise ValueError("nested, sampled, or windowed sources are outside the certified fragment")
    for argument in (
        "qualify",
        "windows",
        "laterals",
        "pivots",
        "sample",
        "with",
        "with_",
        "connect",
        "match",
    ):
        if tree.args.get(argument):
            raise ValueError(f"{argument} is outside the certified fragment")
    if tree.args.get("joins"):
        raise ValueError("joins are outside the certified fragment")
    if tree.args.get("group") is not None:
        raise ValueError("GROUP BY is outside the certified fragment")
    if tree.args.get("having") is not None:
        raise ValueError("HAVING is outside the certified fragment")
    if tree.args.get("limit") is not None or tree.args.get("offset") is not None:
        raise ValueError("LIMIT/OFFSET is outside the certified fragment")
    if tree.args.get("order") is not None:
        raise ValueError("ORDER BY is outside the certified fragment")
    if tree.args.get("distinct") is not None:
        raise ValueError("DISTINCT is outside the certified fragment")
    tables = list(tree.find_all(exp.Table))
    if len(tables) != 1:
        raise ValueError("exactly one base table is required")
    from_clause = tree.args.get("from_") or tree.args.get("from")
    if from_clause is None or not isinstance(from_clause.this, exp.Table):
        raise ValueError("FROM must reference one direct base table")
    table_expression = tables[0]
    if table_expression.args.get("db") or table_expression.args.get("catalog"):
        raise ValueError(
            "schema/catalog-qualified tables are outside the certified fragment"
        )
    table = str(table_expression.name)
    allowed_qualifiers = {table}
    if table_expression.alias:
        allowed_qualifiers.add(str(table_expression.alias))
    for column in tree.find_all(exp.Column):
        qualifier = str(column.table or "")
        if qualifier and qualifier not in allowed_qualifiers:
            raise ValueError("column qualifier does not name the direct base table")
    predicate = _parse_predicate(tree)

    raw_expressions = list(tree.expressions)
    expressions = [_strip_alias(item) for item in raw_expressions]
    aggregate_nodes = [
        item for item in expressions if isinstance(item, (exp.Count, exp.Sum, exp.Avg))
    ]
    if aggregate_nodes:
        if len(expressions) != 1 or len(aggregate_nodes) != 1:
            raise ValueError("one additive aggregate expression is required")
        aggregate_node = aggregate_nodes[0]
        if isinstance(aggregate_node, exp.Count):
            if not isinstance(aggregate_node.this, exp.Star):
                raise ValueError("only COUNT(*) is supported")
            aggregate = "count"
            measure = "*"
        elif isinstance(aggregate_node, exp.Sum):
            aggregate = "sum"
            measure = _column_name(aggregate_node.this)
        else:
            raise ValueError("AVG is not additive without sufficient statistics")
        return QueryDescriptor(
            table=table,
            mode="additive_aggregate",
            output_columns=(),
            aggregate=aggregate,
            measure_column=measure,
            **predicate,
        )

    output_columns: list[str] = []
    for raw_expression, expression in zip(
        raw_expressions, expressions, strict=True
    ):
        if isinstance(raw_expression, exp.Alias):
            raise ValueError("row-set column aliases are outside the certified fragment")
        if not isinstance(expression, exp.Column):
            raise ValueError("row-set SELECT list must contain only columns")
        output_columns.append(str(expression.name))
    if not output_columns:
        raise ValueError("row-set SELECT list must not be empty")
    if len(set(output_columns)) != len(output_columns):
        raise ValueError("duplicate row-set output columns are not supported")
    return QueryDescriptor(
        table=table,
        mode="rowset",
        # Preserve SELECT-list order. Bag equivalence includes column order;
        # sorting here could certify two differently ordered answers as equal.
        output_columns=tuple(output_columns),
        aggregate="",
        measure_column="",
        **predicate,
    )


def _descriptor_record(
    table: str,
    *,
    mode: str,
    outputs: Sequence[str] = (),
    aggregate: str = "",
    measure: str = "",
    predicate_kind: str,
    predicate_column: str,
    lower: float | None = None,
    upper: float | None = None,
    categories: Sequence[str] = (),
) -> QueryDescriptor:
    return QueryDescriptor(
        table=table,
        mode=mode,
        output_columns=tuple(dict.fromkeys(str(item) for item in outputs)),
        aggregate=aggregate,
        measure_column=measure,
        predicate_kind=predicate_kind,
        predicate_column=predicate_column,
        lower=lower,
        upper=upper,
        categories=tuple(sorted(str(item) for item in categories)),
    )


def descriptor_to_sql(descriptor: QueryDescriptor) -> str:
    if descriptor.mode == "rowset":
        select_list = ", ".join(
            _quote_identifier(column) for column in descriptor.output_columns
        )
    elif descriptor.aggregate == "count":
        select_list = "COUNT(*) AS value"
    elif descriptor.aggregate == "sum":
        select_list = f"SUM({_quote_identifier(descriptor.measure_column)}) AS value"
    else:
        raise ValueError(f"Cannot render descriptor mode: {descriptor.mode}")

    query = f"SELECT {select_list} FROM {_quote_identifier(descriptor.table)}"
    if descriptor.predicate_kind == "numeric_interval":
        query += (
            f" WHERE {_quote_identifier(descriptor.predicate_column)}"
            f" >= {_sql_literal(descriptor.lower)}"
            f" AND {_quote_identifier(descriptor.predicate_column)}"
            f" < {_sql_literal(descriptor.upper)}"
        )
    elif descriptor.predicate_kind == "category_set":
        values = ", ".join(_sql_literal(item) for item in descriptor.categories)
        query += f" WHERE {_quote_identifier(descriptor.predicate_column)} IN ({values})"
    elif descriptor.predicate_kind != "all":
        raise ValueError(f"Unsupported predicate kind: {descriptor.predicate_kind}")
    return query


def _add_asset(
    rows: list[dict[str, Any]],
    seen_sql: set[str],
    *,
    dataset: str,
    descriptor: QueryDescriptor | None,
    sql: str,
    family: str,
    template_group: str,
    expected_supported: bool,
) -> str | None:
    normalized = " ".join(sql.split())
    if normalized in seen_sql:
        return None
    seen_sql.add(normalized)
    asset_id = _stable_id("asset", dataset, normalized)
    rows.append(
        {
            "dataset": dataset,
            "asset_id": asset_id,
            "sql": normalized,
            "family": family,
            "template_group": template_group,
            "expected_supported": bool(expected_supported),
            "oracle_descriptor": descriptor.to_json() if descriptor else "",
        }
    )
    return asset_id


def generate_deterministic_workload(
    *,
    dataset: str,
    table_specs: Sequence[Mapping[str, Any]],
    max_assets: int,
    append_negative_queries: bool = True,
) -> pd.DataFrame:
    """Generate diverse SQL assets from real catalog statistics.

    ``table_specs`` entries contain:

    ``table``, ``output_columns``, ``numeric_columns`` (column -> sorted
    boundaries), ``categorical_columns`` (column -> values), and
    ``measure_columns``.
    """

    # ``max_assets`` denotes the active supported workload. A small negative
    # parser suite is appended afterward and does not inflate scale.
    supported_target = max(12, int(max_assets))
    rows: list[dict[str, Any]] = []
    seen_sql: set[str] = set()
    buckets: dict[str, list[list[tuple[QueryDescriptor, str, str]]]] = {}

    def output_sets(columns: Sequence[str]) -> list[tuple[str, ...]]:
        values = []
        for width in (1, 2, 3):
            for start in range(len(columns)):
                selected = tuple(
                    columns[(start + offset) % len(columns)]
                    for offset in range(width)
                )
                values.append(tuple(sorted(set(selected))))
        return list(dict.fromkeys(values))

    def add_group(
        bucket: str,
        descriptors: Sequence[QueryDescriptor],
        target_family: str,
        component_family: str,
        template_group: str,
    ) -> None:
        buckets.setdefault(bucket, []).append(
            [
                (
                    descriptor,
                    target_family if position == 0 else component_family,
                    template_group,
                )
                for position, descriptor in enumerate(descriptors)
            ]
        )

    for spec in table_specs:
        table = str(spec["table"])
        outputs = [str(item) for item in spec["output_columns"]]
        if not outputs:
            raise ValueError(f"{table} must declare output columns")
        measures = [str(item) for item in spec.get("measure_columns", [])]
        table_output_sets = output_sets(outputs)
        numeric = {
            str(column): sorted(set(float(v) for v in values if np.isfinite(v)))
            for column, values in dict(spec.get("numeric_columns", {})).items()
        }
        for predicate_column, boundaries in numeric.items():
            for start in range(len(boundaries) - 2):
                lower, middle, upper = boundaries[start : start + 3]
                if not lower < middle < upper:
                    continue
                intervals = [(lower, upper), (lower, middle), (middle, upper)]
                for selected_outputs in table_output_sets:
                    descriptors = [
                        _descriptor_record(
                            table,
                            mode="rowset",
                            outputs=selected_outputs,
                            predicate_kind="numeric_interval",
                            predicate_column=predicate_column,
                            lower=a,
                            upper=b,
                        )
                        for a, b in intervals
                    ]
                    add_group(
                        f"{table}::numeric_rowset",
                        descriptors,
                        "numeric_partition_target",
                        "numeric_partition_component",
                        (
                            f"{table}::rowset::{predicate_column}::"
                            f"{','.join(selected_outputs)}"
                        ),
                    )
                for aggregate, measure in [
                    ("count", "*"),
                    *[("sum", column) for column in measures],
                ]:
                    descriptors = [
                        _descriptor_record(
                            table,
                            mode="additive_aggregate",
                            aggregate=aggregate,
                            measure=measure,
                            predicate_kind="numeric_interval",
                            predicate_column=predicate_column,
                            lower=a,
                            upper=b,
                        )
                        for a, b in intervals
                    ]
                    add_group(
                        f"{table}::aggregate::{aggregate}",
                        descriptors,
                        "aggregate_partition_target",
                        "aggregate_partition_component",
                        f"{table}::{aggregate}:{measure}::{predicate_column}",
                    )

        categorical = {
            str(column): [str(v) for v in values]
            for column, values in dict(spec.get("categorical_columns", {})).items()
        }
        for predicate_column, categories in categorical.items():
            for left, right in combinations(categories[:12], 2):
                category_sets = [(left, right), (left,), (right,)]
                for selected_outputs in table_output_sets:
                    descriptors = [
                        _descriptor_record(
                            table,
                            mode="rowset",
                            outputs=selected_outputs,
                            predicate_kind="category_set",
                            predicate_column=predicate_column,
                            categories=category_set,
                        )
                        for category_set in category_sets
                    ]
                    add_group(
                        f"{table}::categorical_rowset",
                        descriptors,
                        "categorical_partition_target",
                        "categorical_partition_component",
                        (
                            f"{table}::rowset::{predicate_column}::"
                            f"{','.join(selected_outputs)}"
                        ),
                    )

    active_buckets = [
        (name, groups)
        for name, groups in sorted(buckets.items())
        if groups
    ]
    positions = {name: 0 for name, _ in active_buckets}
    while len(rows) < supported_target and active_buckets:
        made_progress = False
        for name, groups in active_buckets:
            position = positions[name]
            if position >= len(groups):
                continue
            positions[name] += 1
            made_progress = True
            for descriptor, family, template_group in groups[position]:
                if len(rows) >= supported_target:
                    break
                _add_asset(
                    rows,
                    seen_sql,
                    dataset=dataset,
                    descriptor=descriptor,
                    sql=descriptor_to_sql(descriptor),
                    family=family,
                    template_group=template_group,
                    expected_supported=True,
                )
        if not made_progress:
            break
    if len(rows) != supported_target:
        raise ValueError(
            f"Table specifications for {dataset} produced {len(rows)} "
            f"unique supported assets; {supported_target} were requested"
        )

    # Explicit unsupported and near-miss queries exercise fail-closed parsing.
    unsupported_templates: list[tuple[str, str]] = []
    for spec in table_specs:
        table = str(spec["table"])
        outputs = [str(item) for item in spec["output_columns"]]
        measures = [str(item) for item in spec.get("measure_columns", [])]
        if outputs:
            first_output = _quote_identifier(outputs[0])
            quoted_table = _quote_identifier(table)
            unsupported_templates.extend(
                [
                    (
                        f"SELECT {first_output} FROM main.{quoted_table}",
                        "unsupported_schema_qualified_table",
                    ),
                    (
                        f"SELECT {first_output} FROM {quoted_table} "
                        f"UNION ALL SELECT {first_output} FROM {quoted_table}",
                        "unsupported_set_operation",
                    ),
                    (
                        f"WITH q AS (SELECT {first_output} FROM {quoted_table}) "
                        f"SELECT {first_output} FROM q",
                        "unsupported_cte",
                    ),
                ]
            )
        for output in outputs:
            unsupported_templates.extend(
                [
                    (
                        f"SELECT DISTINCT {_quote_identifier(output)} "
                        f"FROM {_quote_identifier(table)}",
                        "unsupported_distinct",
                    ),
                    (
                        f"SELECT {_quote_identifier(output)} "
                        f"FROM {_quote_identifier(table)} LIMIT 10",
                        "unsupported_limit",
                    ),
                ]
            )
        for measure in measures:
            unsupported_templates.append(
                (
                    f"SELECT AVG({_quote_identifier(measure)}) AS value "
                    f"FROM {_quote_identifier(table)}",
                    "unsupported_nonadditive_average",
                )
            )
        categorical = dict(spec.get("categorical_columns", {}))
        for group_column in categorical:
            unsupported_templates.append(
                (
                    f"SELECT {_quote_identifier(group_column)}, COUNT(*) AS n "
                    f"FROM {_quote_identifier(table)} "
                    f"GROUP BY {_quote_identifier(group_column)}",
                    "unsupported_groupby",
                )
            )

    if len(table_specs) >= 2:
        left = table_specs[0]
        right = table_specs[1]
        left_column = str(left["output_columns"][0])
        right_column = str(right["output_columns"][0])
        unsupported_templates.append(
            (
                f"SELECT l.{_quote_identifier(left_column)} "
                f"FROM {_quote_identifier(str(left['table']))} AS l "
                f"JOIN {_quote_identifier(str(right['table']))} AS r "
                f"ON l.{_quote_identifier(left_column)} = "
                f"r.{_quote_identifier(right_column)}",
                "unsupported_join",
            )
        )

    if not unsupported_templates:
        raise ValueError("At least one unsupported query template is required")
    if append_negative_queries:
        for sql, family in unsupported_templates:
            _add_asset(
                rows,
                seen_sql,
                dataset=dataset,
                descriptor=None,
                sql=sql,
                family=family,
                template_group=family,
                expected_supported=False,
            )

    result = pd.DataFrame(rows)
    supported_count = int(result["expected_supported"].sum())
    if supported_count != max_assets:
        raise AssertionError(
            f"Generated {supported_count} supported assets, expected {max_assets}"
        )
    return result


def _same_semantic_surface(left: QueryDescriptor, right: QueryDescriptor) -> bool:
    return (
        left.table == right.table
        and left.mode == right.mode
        and left.output_columns == right.output_columns
        and left.aggregate == right.aggregate
        and left.measure_column == right.measure_column
        and left.predicate_kind == right.predicate_kind
        and left.predicate_column == right.predicate_column
    )


def _predicate_contains(outer: QueryDescriptor, inner: QueryDescriptor) -> bool:
    if (
        outer.predicate_kind != inner.predicate_kind
        or outer.predicate_column != inner.predicate_column
    ):
        return False
    if outer.predicate_kind == "all":
        return True
    if outer.predicate_kind == "numeric_interval":
        return bool(
            outer.lower <= inner.lower
            and inner.upper <= outer.upper
            and (outer.lower, outer.upper) != (inner.lower, inner.upper)
        )
    if outer.predicate_kind == "category_set":
        outer_values = set(outer.categories)
        inner_values = set(inner.categories)
        return bool(inner_values < outer_values)
    return False


def infer_relations_from_descriptors(
    assets: pd.DataFrame,
    descriptors: Mapping[str, QueryDescriptor],
) -> pd.DataFrame:
    """Infer all certified relations for the supported fragment."""

    rows: list[dict[str, Any]] = []
    by_dataset = assets.groupby("dataset", sort=False)
    for dataset, dataset_assets in by_dataset:
        records = [
            (str(row.asset_id), descriptors[str(row.asset_id)])
            for row in dataset_assets.itertuples(index=False)
            if str(row.asset_id) in descriptors
        ]

        # Exact disjoint two-part decompositions.
        groups: dict[tuple[Any, ...], list[tuple[str, QueryDescriptor]]] = {}
        for asset_id, descriptor in records:
            groups.setdefault(descriptor.relation_group_key(), []).append(
                (asset_id, descriptor)
            )
        for _, group in groups.items():
            predicate_index: dict[tuple[Any, ...], list[str]] = {}
            for asset_id, descriptor in group:
                predicate_index.setdefault(descriptor.predicate_key(), []).append(
                    asset_id
                )
            for target_id, target in group:
                component_pairs: set[tuple[str, str]] = set()
                if target.predicate_kind == "numeric_interval":
                    for left_id, left in group:
                        if (
                            left_id == target_id
                            or left.predicate_kind != "numeric_interval"
                            or left.lower != target.lower
                            or not (target.lower < left.upper < target.upper)
                        ):
                            continue
                        right_key = (
                            "numeric_interval",
                            target.predicate_column,
                            left.upper,
                            target.upper,
                            (),
                        )
                        for right_id in predicate_index.get(right_key, []):
                            if right_id != target_id and right_id != left_id:
                                component_pairs.add(tuple(sorted([left_id, right_id])))
                elif target.predicate_kind == "category_set":
                    target_values = set(target.categories)
                    for left_id, left in group:
                        left_values = set(left.categories)
                        if (
                            left_id == target_id
                            or not left_values
                            or not left_values < target_values
                        ):
                            continue
                        right_values = tuple(sorted(target_values - left_values))
                        right_key = (
                            "category_set",
                            target.predicate_column,
                            None,
                            None,
                            right_values,
                        )
                        for right_id in predicate_index.get(right_key, []):
                            if right_id != target_id and right_id != left_id:
                                component_pairs.add(tuple(sorted([left_id, right_id])))
                for components in sorted(component_pairs):
                    relation_id = _stable_id("decomp", dataset, target_id, *components)
                    rows.append(
                        {
                            "dataset": dataset,
                            "relation_id": relation_id,
                            "relation_type": "decomposition",
                            "target_id": target_id,
                            "component_ids": json.dumps(list(components)),
                            "derived_id": "",
                            "source_id": "",
                            "sound": True,
                            "proof_rule": "disjoint_partition_union",
                        }
                    )

        # Projection and filter derivations for row-set queries.
        rowsets = [
            (asset_id, descriptor)
            for asset_id, descriptor in records
            if descriptor.mode == "rowset"
        ]
        projection_groups: dict[
            tuple[Any, ...], list[tuple[str, QueryDescriptor]]
        ] = {}
        filter_groups: dict[
            tuple[str, str], list[tuple[str, QueryDescriptor]]
        ] = {}
        for asset_id, descriptor in rowsets:
            projection_groups.setdefault(
                (descriptor.table, descriptor.predicate_key()), []
            ).append((asset_id, descriptor))
            filter_groups.setdefault(
                (descriptor.table, descriptor.predicate_column), []
            ).append((asset_id, descriptor))

        for projection_group in projection_groups.values():
            for (left_id, left), (right_id, right) in combinations(
                projection_group, 2
            ):
                left_outputs = set(left.output_columns)
                right_outputs = set(right.output_columns)
                if left_outputs < right_outputs and len(right_outputs - left_outputs) == 1:
                    derived_id, source_id = left_id, right_id
                elif right_outputs < left_outputs and len(left_outputs - right_outputs) == 1:
                    derived_id, source_id = right_id, left_id
                else:
                    derived_id = source_id = ""
                if derived_id:
                    relation_id = _stable_id(
                        "projection", dataset, derived_id, source_id
                    )
                    rows.append(
                        {
                            "dataset": dataset,
                            "relation_id": relation_id,
                            "relation_type": "projection_derivation",
                            "target_id": "",
                            "component_ids": "[]",
                            "derived_id": derived_id,
                            "source_id": source_id,
                            "sound": True,
                            "proof_rule": "bag_projection",
                        }
                    )

        for filter_group in filter_groups.values():
            for (left_id, left), (right_id, right) in combinations(filter_group, 2):
                if left.table != right.table:
                    continue
                # A broader row-set answer can be filtered into a narrower one
                # only when it carries every desired output and predicate column.
                for outer_id, outer, inner_id, inner in [
                    (left_id, left, right_id, right),
                    (right_id, right, left_id, left),
                ]:
                    outer_outputs = set(outer.output_columns)
                    inner_outputs = set(inner.output_columns)
                    if (
                        _predicate_contains(outer, inner)
                        and inner_outputs <= outer_outputs
                        and inner.predicate_column in outer_outputs
                    ):
                        relation_id = _stable_id(
                            "filter", dataset, inner_id, outer_id
                        )
                        rows.append(
                            {
                                "dataset": dataset,
                                "relation_id": relation_id,
                                "relation_type": "filter_derivation",
                                "target_id": "",
                                "component_ids": "[]",
                                "derived_id": inner_id,
                                "source_id": outer_id,
                                "sound": True,
                                "proof_rule": "bag_filter_then_projection",
                            }
                        )

    columns = [
        "dataset",
        "relation_id",
        "relation_type",
        "target_id",
        "component_ids",
        "derived_id",
        "source_id",
        "sound",
        "proof_rule",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows, columns=columns)
        .drop_duplicates("relation_id")
        .sort_values(["dataset", "relation_type", "relation_id"])
        .reset_index(drop=True)
    )


def oracle_relations(assets: pd.DataFrame) -> pd.DataFrame:
    """Construct ground-truth relations without calling the SQL compiler.

    This implementation deliberately does not reuse
    :func:`infer_relations_from_descriptors`.  It reads the descriptors retained
    by the deterministic workload constructor and derives relation membership
    through a separate pairwise oracle.  The resulting comparison therefore
    tests both SQL parsing and the compiler's inference path rather than
    comparing that path with itself.
    """

    rows: list[dict[str, Any]] = []

    def add_decomposition(
        dataset: str,
        target_id: str,
        component_ids: Sequence[str],
    ) -> None:
        components = tuple(sorted(str(item) for item in component_ids))
        rows.append(
            {
                "dataset": dataset,
                "relation_id": _stable_id(
                    "oracle_decomp", dataset, target_id, *components
                ),
                "relation_type": "decomposition",
                "target_id": target_id,
                "component_ids": json.dumps(list(components)),
                "derived_id": "",
                "source_id": "",
                "sound": True,
                "proof_rule": "construction_oracle_disjoint_union",
            }
        )

    def add_derivation(
        dataset: str,
        relation_type: str,
        derived_id: str,
        source_id: str,
    ) -> None:
        rows.append(
            {
                "dataset": dataset,
                "relation_id": _stable_id(
                    f"oracle_{relation_type}",
                    dataset,
                    derived_id,
                    source_id,
                ),
                "relation_type": relation_type,
                "target_id": "",
                "component_ids": "[]",
                "derived_id": derived_id,
                "source_id": source_id,
                "sound": True,
                "proof_rule": (
                    "construction_oracle_projection"
                    if relation_type == "projection_derivation"
                    else "construction_oracle_filter"
                ),
            }
        )

    for dataset, dataset_assets in assets.groupby("dataset", sort=False):
        records = [
            (
                str(row.asset_id),
                QueryDescriptor.from_json(str(row.oracle_descriptor)),
            )
            for row in dataset_assets.itertuples(index=False)
            if str(row.oracle_descriptor)
        ]

        # Decomposition ground truth is recovered by joining pairs of
        # components to the exact union descriptor produced by construction.
        decomposition_groups: dict[
            tuple[Any, ...], list[tuple[str, QueryDescriptor]]
        ] = {}
        for asset_id, descriptor in records:
            key = (
                descriptor.table,
                descriptor.mode,
                tuple(descriptor.output_columns),
                descriptor.aggregate,
                descriptor.measure_column,
                descriptor.predicate_kind,
                descriptor.predicate_column,
            )
            decomposition_groups.setdefault(key, []).append(
                (asset_id, descriptor)
            )

        for group in decomposition_groups.values():
            predicate_to_ids: dict[tuple[Any, ...], list[str]] = {}
            for asset_id, descriptor in group:
                key = (
                    descriptor.lower,
                    descriptor.upper,
                    tuple(descriptor.categories),
                )
                predicate_to_ids.setdefault(key, []).append(asset_id)
            for (left_id, left), (right_id, right) in combinations(group, 2):
                if left.predicate_kind == "numeric_interval":
                    ordered = sorted(
                        [(left_id, left), (right_id, right)],
                        key=lambda item: (item[1].lower, item[1].upper),
                    )
                    (_, first), (_, second) = ordered
                    if first.upper != second.lower:
                        continue
                    target_key = (first.lower, second.upper, ())
                elif left.predicate_kind == "category_set":
                    left_values = set(left.categories)
                    right_values = set(right.categories)
                    if (
                        not left_values
                        or not right_values
                        or left_values & right_values
                    ):
                        continue
                    target_key = (
                        None,
                        None,
                        tuple(sorted(left_values | right_values)),
                    )
                else:
                    continue
                for target_id in predicate_to_ids.get(target_key, []):
                    if target_id not in {left_id, right_id}:
                        add_decomposition(
                            dataset,
                            target_id,
                            [left_id, right_id],
                        )

        rowsets = [
            (asset_id, descriptor)
            for asset_id, descriptor in records
            if descriptor.mode == "rowset"
        ]

        # Projection truth: same base relation and predicate, with one output
        # removed.  This code uses construction descriptors, not parsed SQL.
        projection_groups: dict[
            tuple[Any, ...], list[tuple[str, QueryDescriptor]]
        ] = {}
        for asset_id, descriptor in rowsets:
            key = (
                descriptor.table,
                descriptor.predicate_kind,
                descriptor.predicate_column,
                descriptor.lower,
                descriptor.upper,
                tuple(descriptor.categories),
            )
            projection_groups.setdefault(key, []).append(
                (asset_id, descriptor)
            )
        for group in projection_groups.values():
            for (left_id, left), (right_id, right) in combinations(group, 2):
                left_outputs = set(left.output_columns)
                right_outputs = set(right.output_columns)
                if (
                    left_outputs < right_outputs
                    and len(right_outputs - left_outputs) == 1
                ):
                    add_derivation(
                        dataset,
                        "projection_derivation",
                        left_id,
                        right_id,
                    )
                elif (
                    right_outputs < left_outputs
                    and len(left_outputs - right_outputs) == 1
                ):
                    add_derivation(
                        dataset,
                        "projection_derivation",
                        right_id,
                        left_id,
                    )

        # Filter truth is tested by a second containment implementation.
        filter_groups: dict[
            tuple[str, str], list[tuple[str, QueryDescriptor]]
        ] = {}
        for asset_id, descriptor in rowsets:
            filter_groups.setdefault(
                (descriptor.table, descriptor.predicate_column), []
            ).append((asset_id, descriptor))
        for group in filter_groups.values():
            for (left_id, left), (right_id, right) in combinations(group, 2):
                for source_id, source, derived_id, derived in [
                    (left_id, left, right_id, right),
                    (right_id, right, left_id, left),
                ]:
                    if (
                        source.predicate_kind != derived.predicate_kind
                        or source.predicate_column != derived.predicate_column
                    ):
                        continue
                    if source.predicate_kind == "numeric_interval":
                        strictly_contains = bool(
                            source.lower <= derived.lower
                            and derived.upper <= source.upper
                            and (source.lower, source.upper)
                            != (derived.lower, derived.upper)
                        )
                    elif source.predicate_kind == "category_set":
                        strictly_contains = bool(
                            set(derived.categories) < set(source.categories)
                        )
                    else:
                        strictly_contains = False
                    if (
                        strictly_contains
                        and set(derived.output_columns)
                        <= set(source.output_columns)
                        and derived.predicate_column
                        in set(source.output_columns)
                    ):
                        add_derivation(
                            dataset,
                            "filter_derivation",
                            derived_id,
                            source_id,
                        )

    columns = [
        "dataset",
        "relation_id",
        "relation_type",
        "target_id",
        "component_ids",
        "derived_id",
        "source_id",
        "sound",
        "proof_rule",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows, columns=columns)
        .drop_duplicates(
            [
                "dataset",
                "relation_type",
                "target_id",
                "component_ids",
                "derived_id",
                "source_id",
            ]
        )
        .sort_values(["dataset", "relation_type", "relation_id"])
        .reset_index(drop=True)
    )


def compile_relations(
    assets: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Parse every asset, infer relations, and return diagnostics."""

    descriptors: dict[str, QueryDescriptor] = {}
    parse_rows = []
    unsupported_rows = []
    for row in assets.itertuples(index=False):
        try:
            descriptor = descriptor_from_sql(str(row.sql))
            descriptors[str(row.asset_id)] = descriptor
            parse_rows.append(
                {
                    "dataset": row.dataset,
                    "asset_id": row.asset_id,
                    "parsed_supported": True,
                    "reason": "supported",
                    "compiled_descriptor": descriptor.to_json(),
                }
            )
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            parse_rows.append(
                {
                    "dataset": row.dataset,
                    "asset_id": row.asset_id,
                    "parsed_supported": False,
                    "reason": reason,
                    "compiled_descriptor": "",
                }
            )
            unsupported_rows.append(
                {
                    "dataset": row.dataset,
                    "asset_id": row.asset_id,
                    "sql": row.sql,
                    "reason": reason,
                }
            )
    relations = infer_relations_from_descriptors(assets, descriptors)
    return relations, pd.DataFrame(parse_rows), pd.DataFrame(unsupported_rows)


def _relation_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    relation_type = str(row["relation_type"])
    if relation_type == "decomposition":
        components = tuple(sorted(json.loads(str(row["component_ids"]))))
        return relation_type, str(row["target_id"]), components
    return relation_type, str(row["derived_id"]), str(row["source_id"])


def evaluate_compiler(
    assets: pd.DataFrame,
    ground_truth: pd.DataFrame,
    compiled: pd.DataFrame,
    parse_audit: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate relation precision/recall and supported-query classification."""

    detail_rows = []
    metric_rows = []
    for dataset, dataset_assets in assets.groupby("dataset", sort=False):
        truth_rows = ground_truth[ground_truth["dataset"] == dataset]
        compiled_rows = compiled[compiled["dataset"] == dataset]
        truth = {
            _relation_key(row): str(row["relation_id"])
            for row in truth_rows.to_dict(orient="records")
        }
        found = {
            _relation_key(row): str(row["relation_id"])
            for row in compiled_rows.to_dict(orient="records")
        }
        for key in sorted(set(truth) | set(found), key=str):
            detail_rows.append(
                {
                    "dataset": dataset,
                    "relation_key": _canonical_json(key),
                    "in_ground_truth": key in truth,
                    "compiled": key in found,
                    "classification": (
                        "true_positive"
                        if key in truth and key in found
                        else "false_negative"
                        if key in truth
                        else "false_positive"
                    ),
                }
            )
        true_positives = len(set(truth) & set(found))
        false_positives = len(set(found) - set(truth))
        false_negatives = len(set(truth) - set(found))
        precision = (
            true_positives / (true_positives + false_positives)
            if true_positives + false_positives
            else 1.0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if true_positives + false_negatives
            else 1.0
        )
        parse = parse_audit[parse_audit["dataset"] == dataset].merge(
            dataset_assets[["asset_id", "expected_supported"]],
            on="asset_id",
            how="left",
        )
        classification_correct = (
            parse["parsed_supported"].astype(bool)
            == parse["expected_supported"].astype(bool)
        )
        metric_rows.append(
            {
                "dataset": dataset,
                "assets": len(dataset_assets),
                "expected_supported_assets": int(
                    dataset_assets["expected_supported"].sum()
                ),
                "parsed_supported_assets": int(parse["parsed_supported"].sum()),
                "query_classification_accuracy": float(classification_correct.mean()),
                "ground_truth_relations": len(truth),
                "compiled_relations": len(found),
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "precision": precision,
                "recall": recall,
            }
        )
    return pd.DataFrame(metric_rows), pd.DataFrame(detail_rows)


def relation_sql_equivalence_checks(
    *,
    connection,
    assets: pd.DataFrame,
    relations: pd.DataFrame,
    max_relations_per_dataset: int = 200,
) -> pd.DataFrame:
    """Execute deterministic semantic checks for compiled relations.

    Every check uses ``EXCEPT ALL`` in both directions so bag multiplicities are
    tested, not only sets or row counts.
    """

    sql_by_id = dict(zip(assets["asset_id"], assets["sql"]))

    def bag_difference_count(left_sql: str, right_sql: str) -> int:
        query = (
            "SELECT COUNT(*) FROM "
            f"(({left_sql}) EXCEPT ALL ({right_sql})) AS bag_difference"
        )
        return int(connection.execute(query).fetchone()[0])

    def predicate_sql(descriptor: QueryDescriptor, qualifier: str) -> str:
        column = f"{qualifier}.{_quote_identifier(descriptor.predicate_column)}"
        if descriptor.predicate_kind == "numeric_interval":
            return (
                f"{column} >= {_sql_literal(descriptor.lower)} "
                f"AND {column} < {_sql_literal(descriptor.upper)}"
            )
        if descriptor.predicate_kind == "category_set":
            values = ", ".join(_sql_literal(item) for item in descriptor.categories)
            return f"{column} IN ({values})"
        if descriptor.predicate_kind == "all":
            return "TRUE"
        raise ValueError(
            f"Cannot render predicate kind {descriptor.predicate_kind}"
        )

    rows = []
    for dataset, group in relations.groupby("dataset", sort=False):
        relation_types = max(1, group["relation_type"].nunique())
        per_type = max(1, math.ceil(max_relations_per_dataset / relation_types))
        selected = (
            group.sort_values(["relation_type", "relation_id"])
            .groupby("relation_type", sort=False, group_keys=False)
            .head(per_type)
            .head(max_relations_per_dataset)
        )
        for relation in selected.to_dict(orient="records"):
            started = __import__("time").perf_counter_ns()
            if relation["relation_type"] == "decomposition":
                target_sql = sql_by_id[relation["target_id"]]
                target_descriptor = descriptor_from_sql(target_sql)
                component_ids = json.loads(relation["component_ids"])
                components = " UNION ALL ".join(
                    f"SELECT * FROM ({sql_by_id[component_id]}) AS component_{index}"
                    for index, component_id in enumerate(component_ids)
                )
                try:
                    if target_descriptor.mode == "additive_aggregate":
                        target_value = connection.execute(target_sql).fetchone()[0]
                        component_values = [
                            connection.execute(sql_by_id[component_id]).fetchone()[0]
                            for component_id in component_ids
                        ]
                        target_numeric = (
                            0.0 if target_value is None else float(target_value)
                        )
                        component_numeric = sum(
                            0.0 if value is None else float(value)
                            for value in component_values
                        )
                        absolute_error = abs(target_numeric - component_numeric)
                        passed = bool(
                            np.isclose(
                                target_numeric,
                                component_numeric,
                                rtol=1e-10,
                                atol=1e-8,
                            )
                        )
                        missing_left = missing_right = 0 if passed else 1
                        reason = (
                            "additive_aggregate_equality"
                            if passed
                            else f"aggregate_error={absolute_error:.12g}"
                        )
                    else:
                        missing_left = bag_difference_count(target_sql, components)
                        missing_right = bag_difference_count(components, target_sql)
                        passed = missing_left == 0 and missing_right == 0
                        reason = "exact_bag_equality" if passed else "bag_difference"
                except Exception as exc:
                    missing_left = missing_right = -1
                    passed = False
                    reason = f"{type(exc).__name__}: {exc}"
            else:
                derived_sql = sql_by_id[relation["derived_id"]]
                source_sql = sql_by_id[relation["source_id"]]
                descriptor = descriptor_from_sql(derived_sql)
                projected_columns = ", ".join(
                    f"source_answer.{_quote_identifier(column)}"
                    for column in descriptor.output_columns
                )
                candidate_sql = (
                    f"SELECT {projected_columns} "
                    f"FROM ({source_sql}) AS source_answer"
                )
                if relation["relation_type"] == "filter_derivation":
                    candidate_sql += (
                        " WHERE "
                        + predicate_sql(descriptor, "source_answer")
                    )
                try:
                    missing_left = bag_difference_count(
                        derived_sql, candidate_sql
                    )
                    missing_right = bag_difference_count(
                        candidate_sql, derived_sql
                    )
                    passed = missing_left == 0 and missing_right == 0
                    reason = "exact_bag_equality" if passed else "bag_difference"
                except Exception as exc:
                    missing_left = missing_right = -1
                    passed = False
                    reason = f"{type(exc).__name__}: {exc}"
            rows.append(
                {
                    "dataset": dataset,
                    "relation_id": relation["relation_id"],
                    "relation_type": relation["relation_type"],
                    "passed": bool(passed),
                    "reason": reason,
                    "left_only_rows": missing_left,
                    "right_only_rows": missing_right,
                    "runtime_ms": (
                        __import__("time").perf_counter_ns() - started
                    )
                    / 1e6,
                }
            )
    return pd.DataFrame(rows)
