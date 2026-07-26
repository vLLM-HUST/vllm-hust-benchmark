# Trend coverage schema

This document is the maintenance reference for
[`schemas/leaderboard_trend_v1.schema.json`](../schemas/leaderboard_trend_v1.schema.json). The
machine-readable schema is authoritative for shape, types, enums, and single-entry conditional
requirements. The semantic rules below explain how those fields are intended to be used by
producers, validators, publishers, and consumers.

The contract version is `trend-coverage/v1`, carried in the entry field `trend_schema_version`. A
producer must not infer these values from an artifact directory name, filename, or frontend state.

## Fields

| Field                  | Type / values                                               | Meaning and usage                                                                                                                                                                              |
| ---------------------- | ----------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `trend_schema_version` | string, `trend-coverage/v1`                                 | Identifies this extension. It is required for trend-aware entries.                                                                                                                             |
| `coverage_class`       | `full-matrix`, `targeted-pair`, `experimental`              | Declares the coverage model. `full-matrix` is a fixed-stack matrix; `targeted-pair` is one side of a baseline/head comparison; `experimental` is outside the formal matrix or comparison.      |
| `campaign_id`          | non-empty string                                            | Groups a coordinated run campaign with a fixed stack and provenance. Required for `full-matrix` and `targeted-pair`; optional for `experimental`. Use `<campaign-name>/<campaign-version>`.    |
| `comparison_id`        | non-empty string                                            | Stable identity of a targeted baseline/head scope. Required for `targeted-pair`, forbidden for `experimental`. It does not prove that both sides exist; T09 must check that.                   |
| `point_role`           | `baseline`, `head`, `checkpoint`, or `null`                 | Role within the declared coverage. `full-matrix` requires `checkpoint`; `targeted-pair` requires `baseline` or `head`; `experimental` must omit it or use `null`.                              |
| `repeat_group`         | non-empty string                                            | Stable identity of one repeated series. Use the campaign and series dimensions, for example `<campaign>::<model>::<hardware>::<precision>::<workload>::<topology>::<engine>`.                  |
| `repeat_index`         | integer ≥ 0                                                 | Zero-based run index within `repeat_group`. Required whenever `repeat_group` is present. Uniqueness and continuity are cross-entry checks.                                                     |
| `canonical_aggregate`  | object                                                      | Published aggregate for a repeated series. Required when `repeat_group` is present and the coverage is not experimental. It must declare a method, count, metric values, and outlier handling. |
| `trend_status`         | `default`, `experimental`, `blocked`, `invalid`, `excluded` | Final publication status. Producers may propose it, but the validator is allowed to downgrade or replace it.                                                                                   |
| `trend_reason`         | non-empty string                                            | Human-readable, actionable explanation for `blocked`, `invalid`, or `excluded`; also required for non-default experimental examples. Keep the original artifact for auditability.              |

## Coverage and status rules

### Full matrix

Use `coverage_class: "full-matrix"`, `point_role: "checkpoint"`, and a `campaign_id`. A repeated
formal series also carries `repeat_group`, `repeat_index`, and `canonical_aggregate`. The schema
validates each entry; T09 must decide whether the campaign has complete model/workload/topology
coverage and whether the repeated series meets the minimum sample policy.

### Targeted pair

Use `coverage_class: "targeted-pair"`, a `campaign_id`, a `comparison_id`, and
`point_role: "baseline"` or `"head"`. A half-pair remains in the raw provenance with
`trend_status: "blocked"` and a reason. T09 must require one comparable baseline and one comparable
head for `default`; JSON Schema cannot compare two separate entries.

### Experimental

Use `coverage_class: "experimental"` for exploratory or not-yet-admitted data. It cannot participate
in a targeted pair, so `comparison_id` is absent and `point_role` is absent or `null`. Its status is
normally `experimental`; `invalid` and `excluded` are allowed for retained audit records such as an
invalid metric or retired data.

### Invalid metrics and exclusion

Schema validation checks numeric shape and ranges, but it does not decide whether a measurement is
physically credible. T09 must apply the metric policy: preserve the raw artifact, set the published
metric to `null` when appropriate, and use `invalid` or `blocked` with an actionable reason.

`excluded` is for retained provenance that must not enter either the default or experimental view,
such as retired records. It is not a synonym for `experimental`.

## Canonical aggregate semantics

`canonical_aggregate.method` is one of `mean`, `median`, `trimmed_mean`, `min`, or `max`.
`trimmed_mean` requires `trim_percent`; mean-like methods require at least two samples in this
schema. `count` records the source sample count, including samples removed or capped by outlier
handling.

Each aggregate metric has a required `value` and may carry `min`, `max`, and `std`. The producer or
aggregate validator must keep the top-level metric value synchronized with the aggregate metric
value. Raw repetitions must not be overwritten.

## Legacy compatibility and ownership

An entry without `trend_schema_version`, `coverage_class`, and `trend_status` is accepted by the
legacy branch so historical artifacts can be read. It is not trend-admitted: T09 must classify
missing coverage as `unknown`/`excluded` (or another explicitly documented migration result) and
must write a new trend-aware entry rather than silently upgrading the old artifact.

The schema owns local structure and conditional field requirements. T09 owns cross-entry pair/matrix
completeness, comparability, repeat-index uniqueness, aggregation consistency, and final admission.
T10/website consumers must use `trend_status` for filtering and must not reconstruct admission
heuristics.

When adding a field, update this document, the JSON Schema, at least one valid and one invalid
fixture where applicable, and the focused schema test in the same change.
