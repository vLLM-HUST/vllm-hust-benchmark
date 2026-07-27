"""Repeat-run aggregation for leaderboard entries.

Converts a series of repeated benchmark runs (same repeat_group) into a
deterministic, auditable canonical_aggregate — eliminating the need for
front-end best-of selection.

Design
------
- Entries are grouped by ``repeat_group`` (a string uniquely identifying
  the campaign + series signature).
- Within each group the entries are sorted by ``repeat_index`` for
  deterministic processing.
- Per-metric statistics are computed (mean, median, trimmed mean, min,
  latest) as well as range (min/max) and dispersion (std).
- Outliers can be detected via IQR or 3σ rules and either removed or capped.
- The result is a ``canonical_aggregate`` object (per T06 §3.6) that is
  atomically paired with the entry's top-level metrics.

  The canonical_aggregate is embedded into the **last** entry of the group
  (sorted by ``repeat_index``) so the group can be represented as a single
  snapshot row. The original raw entries remain on disk unchanged.

Conventions
-----------
- All functions accept ``list[dict]`` of leaderboard entry dicts.
- All functions are **pure** — they never modify their inputs.
- All numeric results are Python ``float`` or ``int``, JSON-serializable.
- Determinism guarantee: same input entries produce the same aggregate
  regardless of iteration order, insertion order, or host platform
  (within ±1e-9 floating-point tolerance).
"""

from __future__ import annotations

import json
import math
import statistics
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_AGG_METHODS = frozenset({"mean", "median", "trimmed_mean", "min", "latest"})
"""Accepted values for ``canonical_aggregate.method``."""

VALID_OUTLIER_HANDLING = frozenset({"none", "removed", "capped"})
"""Accepted values for ``canonical_aggregate.outlier_handling``."""

TRIM_MEAN_MIN_COUNT = 4
"""``trimmed_mean`` requires at least this many entries."""

OUTLIER_REMOVAL_MIN_FRACTION = 2.0 / 3.0
"""After removal, remaining entries must be at least this fraction of original."""

IQR_MULTIPLIER = 1.5
"""Multiplier for IQR outlier detection (Tukey's fences)."""

SIGMA_MULTIPLIER = 3.0
"""Multiplier for 3σ outlier detection."""


# ---------------------------------------------------------------------------
# Series signature helpers
# ---------------------------------------------------------------------------


def build_series_signature(entry: dict[str, Any]) -> str:
    """Return the series signature of an entry from its intrinsic fields.

    This is the string used to identify entries that belong to the same
    series (same model, hardware, precision, workload, chip-count, config,
    engine, engine_version).  It does **not** incorporate campaign_id or
    repeat_group — it is the raw identity that *would* go into a
    repeat_group if one were assigned.

    Returns

        A pipe-delimited string or ``""`` if the entry lacks the required
        fields.
    """
    try:
        model = entry.get("model") or {}
        hardware = entry.get("hardware") or {}
        workload = entry.get("workload") or {}
        parts = [
            str(model.get("canonical_id") or ""),
            str(hardware.get("chip_model") or ""),
            str(model.get("precision") or ""),
            str(workload.get("name") or ""),
            str(hardware.get("chip_count") or ""),
            str(entry.get("config_type") or ""),
            str(entry.get("engine") or ""),
            str(entry.get("engine_version") or ""),
        ]
        sig = "|".join(parts)
        return sig if sig.strip("|") else ""
    except (TypeError, AttributeError):
        return ""


def get_repeat_group(entry: dict[str, Any]) -> str | None:
    """Return the entry's repeat_group string, or ``None``."""
    rg = entry.get("repeat_group")
    if isinstance(rg, str) and rg.strip():
        return rg.strip()
    return None


def get_repeat_index(entry: dict[str, Any]) -> int | None:
    """Return the entry's repeat_index, or ``None``."""
    ri = entry.get("repeat_index")
    if isinstance(ri, int):
        return ri
    if isinstance(ri, float) and ri == int(ri):
        return int(ri)
    return None


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------


def group_entries_by_repeat_group(
    entries: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group entries by their ``repeat_group`` field.

    Entries without a repeat_group are collected under the key ``""``
    (empty string).  Within each group, entries are sorted by
    ``repeat_index`` (ascending); entries without an index sort last
    among themselves, keeping their original relative order
    (stable sort).
    """
    groups: dict[str, list[dict[str, Any]]] = OrderedDict()
    for entry in entries:
        rg = get_repeat_group(entry) or ""
        groups.setdefault(rg, []).append(entry)

    for rg in groups:
        _sort_entries_in_place(groups[rg])
    return groups


def _sort_entries_in_place(entries: list[dict[str, Any]]) -> None:
    """Sort *entries* by repeat_index ascending; None-index items sort last."""
    entries.sort(key=_entry_sort_key)


def _entry_sort_key(entry: dict[str, Any]) -> tuple[int, int]:
    ri = get_repeat_index(entry)
    if ri is not None:
        return (0, ri)
    return (1, 0)


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------


def detect_outliers_iqr(values: list[float]) -> tuple[list[int], tuple[float, float]]:
    """IQR-based outlier detection (Tukey's fences).

    Returns
        ``(outlier_indices, (lower_bound, upper_bound))``.
    """
    if len(values) < 4:
        return [], (float("-inf"), float("inf"))

    sorted_vals = sorted(values)
    n = len(sorted_vals)
    q1 = _percentile(sorted_vals, 25)
    q3 = _percentile(sorted_vals, 75)
    iqr = q3 - q1
    lower = q1 - IQR_MULTIPLIER * iqr
    upper = q3 + IQR_MULTIPLIER * iqr

    outlier_indices = [i for i, v in enumerate(values) if v < lower or v > upper]
    return outlier_indices, (lower, upper)


def detect_outliers_3sigma(
    values: list[float],
) -> tuple[list[int], tuple[float, float]]:
    """3σ-based outlier detection.

    Returns
        ``(outlier_indices, (lower_bound, upper_bound))``.
    """
    if len(values) < 3:
        return [], (float("-inf"), float("inf"))

    mu = statistics.mean(values)
    sigma = statistics.stdev(values) if len(values) > 1 else 0.0
    lower = mu - SIGMA_MULTIPLIER * sigma
    upper = mu + SIGMA_MULTIPLIER * sigma

    outlier_indices = [i for i, v in enumerate(values) if v < lower or v > upper]
    return outlier_indices, (lower, upper)


def _percentile(sorted_values: list[float], percentile: float) -> float:
    """Linear-interpolation percentile (same as numpy default)."""
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    k = (percentile / 100.0) * (n - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)


# ---------------------------------------------------------------------------
# Per-metric statistics
# ---------------------------------------------------------------------------


def compute_metric_stats(
    values: list[float],
    *,
    method: str = "mean",
    trim_percent: float = 0.0,
) -> dict[str, Any]:
    """Compute aggregate statistics for a single metric.

    Parameters
    ----------
    values:
        Raw metric values across repetitions.
    method:
        Aggregation method (must be in ``VALID_AGG_METHODS``).
    trim_percent:
        Fraction to trim from each end (only used when
        ``method="trimmed_mean"``).

    Returns
        A dict conforming to ``canonical_aggregate.metrics.<name>``:

        .. code:: python

            {
                "value": 294.76,
                "min": 280.0,
                "max": 310.0,
                "std": 15.3,
            }

        ``std`` is omitted when only one value is present.
    """
    if not values:
        raise ValueError("Cannot compute stats on empty values list")

    clean = [float(v) for v in values]
    n = len(clean)

    result: dict[str, Any] = {
        "min": min(clean),
        "max": max(clean),
    }
    if n > 1:
        result["std"] = statistics.stdev(clean)

    if method == "mean":
        result["value"] = statistics.mean(clean)
    elif method == "median":
        result["value"] = statistics.median(clean)
    elif method == "trimmed_mean":
        result["value"] = _trimmed_mean(clean, trim_percent)
    elif method == "min":
        result["value"] = min(clean)
    elif method == "latest":
        # Takes the value from the highest repeat_index entry.  The caller
        # must pass values already sorted by repeat_index ascending so that
        # ``clean[-1]`` is the latest entry's raw value.
        result["value"] = clean[-1]
    else:
        raise ValueError(f"Unknown aggregation method: {method!r}")

    return result


def _trimmed_mean(values: list[float], trim_percent: float) -> float:
    """Compute trimmed (truncated) mean.

    ``trim_percent`` is the fraction *per side* (e.g. 0.1 removes 10% from
    each end).  At least one value must remain after trimming.
    """
    if not 0.0 <= trim_percent < 0.5:
        raise ValueError(f"trim_percent must be in [0, 0.5), got {trim_percent}")
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    k = int(math.floor(n * trim_percent))
    # With trim_percent < 0.5, k <= floor(n/2) - 1 so at least 1 value stays
    return statistics.mean(sorted_vals[k : n - k])


# ---------------------------------------------------------------------------
# Outlier handling helpers
# ---------------------------------------------------------------------------


def _handle_outliers(
    values: list[float],
    outlier_handling: str,
    outlier_indices: list[int],
    *,
    detection_method: str = "iqr",
) -> tuple[list[float], str | None]:
    """Apply outlier handling strategy and return (cleaned_values, detail_note).

    Parameters
    ----------
    detection_method:
        Used only in the detail string for audit traceability.
    """
    if outlier_handling == "none" or not outlier_indices:
        return values, None

    outlier_set = set(outlier_indices)
    kept = [v for i, v in enumerate(values) if i not in outlier_set]

    if outlier_handling == "removed":
        if len(kept) < max(1, int(len(values) * OUTLIER_REMOVAL_MIN_FRACTION)):
            # If removal drops too many, fall back to capping
            return _cap_outliers(values, outlier_indices)
        label = "3σ" if detection_method == "3sigma" else "IQR"
        detail = (
            f"{label} method: removed {len(outlier_indices)} outlier(s) "
            f"indices {outlier_indices} from {len(values)} entries. "
            f"Recalculated with n={len(kept)}."
        )
        return kept, detail

    if outlier_handling == "capped":
        return _cap_outliers(values, outlier_indices)

    raise ValueError(f"Unknown outlier_handling: {outlier_handling!r}")


def _cap_outliers(
    values: list[float],
    outlier_indices: list[int],
) -> tuple[list[float], str]:
    """Cap outliers to the nearest non-outlier bound."""
    outlier_set = set(outlier_indices)
    kept = [v for i, v in enumerate(values) if i not in outlier_set]
    if not kept:
        return values, "All values were outliers; capping not possible, kept original"
    lower = min(kept)
    upper = max(kept)
    capped = [
        lower
        if i in outlier_set and v < lower
        else upper
        if i in outlier_set and v > upper
        else v
        for i, v in enumerate(values)
    ]
    detail = (
        f"Capped {len(outlier_indices)} outlier(s) indices {outlier_indices} "
        f"to range [{lower}, {upper}]."
    )
    return capped, detail


# ---------------------------------------------------------------------------
# Validation functions
# ---------------------------------------------------------------------------


def validate_aggregate_method(method: str, count: int) -> list[str]:
    """Validate aggregation method constraints.

    Returns a list of error messages (empty = valid).
    """
    errors: list[str] = []
    if method not in VALID_AGG_METHODS:
        errors.append(
            f"Invalid method {method!r}. Must be one of: {sorted(VALID_AGG_METHODS)}"
        )
        return errors
    if method == "trimmed_mean" and count < TRIM_MEAN_MIN_COUNT:
        errors.append(
            f"trimmed_mean requires count >= {TRIM_MEAN_MIN_COUNT}, got {count}"
        )
    if method in ("min", "latest") and count < 1:
        errors.append(f"count must be >= 1 for {method}, got {count}")
    if method == "mean" and count < 1:
        errors.append(f"count must be >= 1 for mean, got {count}")
    return errors


def validate_aggregate_structure(aggregate: dict[str, Any]) -> list[str]:
    """Validate a canonical_aggregate object structure.

    Returns a list of error messages (empty = valid).
    """
    errors: list[str] = []
    if not isinstance(aggregate, dict):
        return ["canonical_aggregate must be a dict"]

    method = aggregate.get("method")
    if method not in VALID_AGG_METHODS:
        errors.append(
            f"Invalid method {method!r}. Must be one of: {sorted(VALID_AGG_METHODS)}"
        )

    count = aggregate.get("count")
    if not isinstance(count, int) or count < 1:
        errors.append(f"count must be a positive integer, got {count!r}")

    if method == "trimmed_mean":
        tp = aggregate.get("trim_percent", 0.0)
        if not isinstance(tp, (int, float)) or not 0.0 <= tp < 0.5:
            errors.append(f"trim_percent must be in [0, 0.5), got {tp!r}")

    metrics = aggregate.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        errors.append("metrics must be a non-empty dict")
    elif isinstance(metrics, dict):
        for metric_name, mvalue in metrics.items():
            if not isinstance(mvalue, dict):
                errors.append(f"metrics.{metric_name} must be a dict")
                continue
            if "value" not in mvalue:
                errors.append(f"metrics.{metric_name}.value is required")

    outlier = aggregate.get("outlier_handling", "none")
    if outlier not in VALID_OUTLIER_HANDLING:
        errors.append(
            f"Invalid outlier_handling {outlier!r}. Must be: {sorted(VALID_OUTLIER_HANDLING)}"
        )

    return errors


# ---------------------------------------------------------------------------
# Main aggregation logic
# ---------------------------------------------------------------------------


def _collect_metric_values(
    entries: list[dict[str, Any]],
) -> dict[str, list[float]]:
    """Collect all metric values across entries, keyed by metric name.

    Only numeric metrics that appear in *every* entry of the group are
    included (metrics that are ``None`` in any entry are skipped).
    """
    if not entries:
        return {}

    metric_names: set[str] | None = None
    all_names: set[str] = set()
    for entry in entries:
        m = entry.get("metrics")
        if not isinstance(m, dict):
            return {}
        keys = {
            k
            for k, v in m.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v is not None
        }
        all_names.update(m.keys())
        if metric_names is None:
            metric_names = keys
        else:
            metric_names &= keys

    if metric_names is not None:
        dropped = all_names - metric_names
        if dropped:
            warnings.warn(
                f"Dropped {len(dropped)} metric(s) not present as numeric in all entries: "
                f"{sorted(dropped)}. "
                f"Only metrics common to every entry are included in the aggregate.",
                stacklevel=2,
            )

    if not metric_names:
        return {}

    result: dict[str, list[float]] = {k: [] for k in metric_names}
    for entry in entries:
        m = entry.get("metrics", {})
        for k in metric_names:
            result[k].append(float(m[k]))
    return result


def compute_canonical_aggregate(
    entries: list[dict[str, Any]],
    *,
    method: str = "mean",
    trim_percent: float = 0.0,
    outlier_handling: str = "none",
    outlier_detection: str = "iqr",
) -> dict[str, Any]:
    """Compute a complete canonical_aggregate for a group of entries.

    Parameters
    ----------
    entries:
        All entries belonging to the same ``repeat_group``.  They will be
        sorted by ``repeat_index`` internally.
    method:
        Aggregation method (default ``"mean"``).
    trim_percent:
        Per-side trim fraction for ``trimmed_mean`` (default ``0.0``).
    outlier_handling:
        How to treat outliers: ``"none"``, ``"removed"``, or ``"capped"``.
    outlier_detection:
        Outlier detection algorithm: ``"iqr"`` or ``"3sigma"``.

    Returns
        A full canonical_aggregate dict ready to embed in an entry.
    """
    entries = list(entries)
    _sort_entries_in_place(entries)

    if not entries:
        raise ValueError("Cannot compute aggregate for empty entries list")

    method = method or "mean"
    outlier_handling = outlier_handling or "none"
    outlier_detection = outlier_detection or "iqr"

    # Validate
    method_errors = validate_aggregate_method(method, len(entries))
    if method_errors:
        raise ValueError("; ".join(method_errors))

    if outlier_handling not in VALID_OUTLIER_HANDLING:
        raise ValueError(
            f"Invalid outlier_handling {outlier_handling!r}. "
            f"Must be one of: {sorted(VALID_OUTLIER_HANDLING)}"
        )

    # Collect metric values
    metric_values = _collect_metric_values(entries)
    if not metric_values:
        raise ValueError("No common numeric metrics found across entries")

    # Detect outliers per metric (union across all metrics)
    all_outlier_indices: set[int] = set()
    for values in metric_values.values():
        if outlier_detection == "iqr":
            indices, _ = detect_outliers_iqr(values)
        elif outlier_detection == "3sigma":
            indices, _ = detect_outliers_3sigma(values)
        else:
            raise ValueError(
                f"Unknown outlier_detection {outlier_detection!r}. "
                f"Use 'iqr' or '3sigma'."
            )
        all_outlier_indices.update(indices)

    outlier_indices_sorted = sorted(all_outlier_indices)

    # Handle outliers
    agg_metrics: dict[str, dict[str, Any]] = {}
    outlier_detail: str | None = None

    for metric_name, raw_values in metric_values.items():
        if method == "latest":
            # "latest" takes the raw metrics from the highest-repeat_index
            # entry; outlier handling does not apply.
            cleaned = raw_values
        elif outlier_handling != "none" and all_outlier_indices:
            cleaned, detail = _handle_outliers(
                raw_values,
                outlier_handling,
                outlier_indices_sorted,
                detection_method=outlier_detection,
            )
            if detail and outlier_detail is None:
                outlier_detail = detail
        else:
            cleaned = raw_values

        agg_metrics[metric_name] = compute_metric_stats(
            cleaned,
            method=method,
            trim_percent=trim_percent,
        )

    note = (
        f"Aggregated from {len(entries)} independent run(s) "
        f"using {method}"
        + (f" (trim={trim_percent})" if method == "trimmed_mean" else "")
        + "."
    )

    aggregate: dict[str, Any] = {
        "method": method,
        "count": len(entries),
        "metrics": agg_metrics,
        "outlier_handling": outlier_handling,
    }

    if method == "trimmed_mean":
        aggregate["trim_percent"] = trim_percent

    if outlier_detail is not None:
        aggregate["outlier_details"] = outlier_detail
    else:
        aggregate["outlier_details"] = None

    aggregate["note"] = note

    return aggregate


# ---------------------------------------------------------------------------
# Applying aggregate to an entry
# ---------------------------------------------------------------------------


def apply_aggregate_to_entry(
    entry: dict[str, Any],
    aggregate: dict[str, Any],
) -> dict[str, Any]:
    """Return a *new* entry with ``canonical_aggregate`` set and top-level
    ``metrics`` overwritten with the aggregate values.

    The original entry dict is never modified.
    """
    entry = dict(entry)
    metrics = dict(entry.get("metrics", {}) or {})
    agg_metrics = aggregate.get("metrics", {})

    for metric_name, agg_value in agg_metrics.items():
        if isinstance(agg_value, dict) and "value" in agg_value:
            metrics[metric_name] = agg_value["value"]

    entry["metrics"] = metrics
    entry["canonical_aggregate"] = aggregate
    return entry


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------


def aggregate_entries(
    entries: list[dict[str, Any]],
    *,
    method: str = "mean",
    trim_percent: float = 0.0,
    outlier_handling: str = "none",
    outlier_detection: str = "iqr",
) -> list[dict[str, Any]]:
    """Aggregate repeated-run entries into canonical entries.

    This is the top-level entry point.  It:

    1. Groups entries by ``repeat_group``.
    2. For each group with >= 1 entry, computes a canonical_aggregate.
    3. Returns a list of **aggregated entries** — one per group — with
       ``canonical_aggregate`` embedded and top-level ``metrics`` set to
       the aggregate values.

    Entries that lack a ``repeat_group`` are returned verbatim (no
    aggregation applied).

    Parameters
    ----------
    entries:
        All leaderboard entries to process.
    method:
        Aggregation method (default ``"mean"``).
    trim_percent:
        Per-side trim for ``trimmed_mean``.
    outlier_handling:
        ``"none"``, ``"removed"``, or ``"capped"``.
    outlier_detection:
        ``"iqr"`` or ``"3sigma"``.

    Returns
        Aggregated entries, one per repeat_group, with canonical_aggregate.
    """
    groups = group_entries_by_repeat_group(entries)
    result: list[dict[str, Any]] = []

    for rg, group_entries in groups.items():
        if not rg:
            # No repeat_group → pass through unchanged
            result.extend(group_entries)
            continue

        aggregate = compute_canonical_aggregate(
            group_entries,
            method=method,
            trim_percent=trim_percent,
            outlier_handling=outlier_handling,
            outlier_detection=outlier_detection,
        )

        # Apply to the last entry (highest repeat_index) to preserve the
        # entry with the most recent metadata.
        target_entry = apply_aggregate_to_entry(group_entries[-1], aggregate)

        if method == "latest":
            selected_index = get_repeat_index(group_entries[-1])
            all_indices = [get_repeat_index(e) for e in group_entries]
            discarded = sorted(
                idx for idx in all_indices if idx is not None and idx != selected_index
            )
            metadata = dict(target_entry.get("metadata") or {})
            metadata["aggregate_audit"] = {
                "method": "latest",
                "discarded_repeat_indices": discarded,
            }
            target_entry["metadata"] = metadata

        result.append(target_entry)

    return result


# ---------------------------------------------------------------------------
# File-level I/O
# ---------------------------------------------------------------------------


def load_entries_from_paths(paths: list[str]) -> list[dict[str, Any]]:
    """Load leaderboard entry dicts from JSON file paths.

    Each file must contain a single JSON object (one entry).
    """
    entries: list[dict[str, Any]] = []
    for path in paths:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Entry file not found: {path}")
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{path}: must be a JSON object")
        entries.append(data)
    return entries


def write_aggregated_entries(
    entries: list[dict[str, Any]],
    output_dir: str,
    *,
    suffix: str = "_aggregated",
) -> list[str]:
    """Write aggregated entries to *output_dir*.

    Each entry is written as a separate JSON file named
    ``<entry_id>{suffix}.json``.  The original ``entry_id`` is preserved.
    Original files are never touched.

    If an entry lacks an ``entry_id``, a counter-based filename is used
    to avoid collisions.

    Returns a list of written paths.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    seen_ids: set[str] = set()
    nameless_counter = 0
    for entry in entries:
        entry_id = entry.get("entry_id")
        if entry_id and isinstance(entry_id, str) and entry_id.strip():
            base = entry_id.strip()
        else:
            base = f"entry_no_id_{nameless_counter}"
            nameless_counter += 1

        # Avoid accidental collision if two entries share the same entry_id
        if base in seen_ids:
            base = f"{base}_dup"
        seen_ids.add(base)

        filename = f"{base}{suffix}.json"
        path = out / filename
        path.write_text(
            json.dumps(entry, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        written.append(str(path))
    return written
