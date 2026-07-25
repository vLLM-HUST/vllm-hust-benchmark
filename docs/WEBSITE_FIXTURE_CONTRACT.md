# Website fixture contract

> **For**: Website consumers that render leaderboard trend data  
> **Contract version**: `trend-coverage/v1`  
> **Related**: `docs/TREND_COVERAGE_SCHEMA.md`, `tests/test_website_contract.py`

This document defines the consumer-facing contract between the benchmark data
pipeline and the website. It describes how the website should interpret each
trend status, which entries to show as formal trends, and how to use the
provided fixtures for testing.

---

## 1. Six fixture categories

The following six fixtures in `tests/fixtures/trend_coverage/valid/` cover
every data scenario the website must handle. Each fixture file is loadable as
a single JSON object or array and passes JSON Schema validation under
`trend-coverage/v1`.

| # | Category | Fixture file | Produced `trend_status` | Website rendering |
|---|----------|-------------|------------------------|-------------------|
| 1 | **full-matrix** | `full-matrix.json` | `blocked` (without raw repeats) | Hide from trend lines; show diagnostic |
| 2 | **complete targeted pair** | `complete-pair.json` | `default` (all entries) | Plot as paired trend lines |
| 3 | **blocked half-pair** | `blocked-half-pair.json` | `blocked` (all entries) | Hide from trend lines; show diagnostic |
| 4 | **experimental** | `experimental.json` | `experimental` | Show in experimental section only; never in default trend |
| 5 | **invalid metric** | `invalid.json` | `blocked` | Hide from trend lines; show diagnostic |
| 6 | **repeat aggregate** | `repeat-aggregate.json` | `default` (all entries) | Plot as formal trend line with aggregate markers |

### 1.1 Status → rendering matrix

| `trend_status` | Show as default trend? | Show in experimental section? | Show in blocked diagnostics? | Show in excluded list? |
|---------------|----------------------|------------------------------|----------------------------|----------------------|
| `default`     | ✅ Yes               | ❌ No                        | ❌ No                       | ❌ No                |
| `experimental`| ❌ No                | ✅ Yes (separate)             | ❌ No                       | ❌ No                |
| `blocked`     | ❌ No                | ❌ No                        | ✅ Yes (with reason)         | ❌ No                |
| `invalid`     | ❌ No                | ❌ No                        | ✅ Yes (with reason)         | ❌ No                |
| `excluded`    | ❌ No                | ❌ No                        | ❌ No                       | ✅ Yes (provenance) |

---

## 2. Filtering contract

The website MUST NOT show `experimental`, `blocked`, `invalid`, or `excluded`
entries as default formal trend lines. The ONLY allowed filter for the primary
trend chart is:

```python
entries_by_status(entries, "default")
```

Or, expressed declaratively:

```python
trend_entries = [e for e in all_entries if e.get("trend_status") == "default"]
```

### 2.1 Experimental section

Experimental entries (`trend_status == "experimental"`) should be displayed in a
separate section below the main trend chart, clearly labeled as "Experimental"
or "Preview". They must not be mixed into the same axes as formal `default`
trends.

### 2.2 Blocked diagnostics

Blocked entries (`trend_status == "blocked"`) carry a `trend_reason` field with
actionable diagnostic information. The website should expose this in a
diagnostics panel or tooltip. Example reasons:

```
"No comparable baseline entry for comparison_id=..."
"canonical_aggregate.count=3 but repeat_group=... contains 1 raw entries"
"random-latency may omit throughput_tps; keep the sanitized metric null"
```

### 2.3 Invalid entries

Invalid entries (`trend_status == "invalid"`) have critical metric failures
and should be hidden from all trend views. They may be shown in a data-quality
dashboard with their `trend_reason`.

### 2.4 Excluded entries

Excluded entries (`trend_status == "excluded"`) are legacy records retained
for provenance only. They must not appear in any published view.

---

## 3. Using fixtures in website tests

### 3.1 Loading fixtures

The fixtures are JSON files that can be loaded in any language:

```python
import json
entries = json.loads(path.read_text())
# entries is either a dict (single entry) or list[dict]
if not isinstance(entries, list):
    entries = [entries]
```

### 3.2 Testing rendering logic

```python
def test_default_entries_are_plotted():
    entries = load_fixture("complete-pair.json")
    default = [e for e in entries if e["trend_status"] == "default"]
    render_trend_chart(default)
    assert chart.has_series("baseline")
    assert chart.has_series("head")

def test_experimental_is_not_on_main_chart():
    entries = load_fixture("experimental.json")
    default = [e for e in entries if e["trend_status"] == "default"]
    assert len(default) == 0  # never shows on main chart

def test_blocked_half_pair_not_on_trend():
    entries = load_fixture("blocked-half-pair.json")
    default = [e for e in entries if e["trend_status"] == "default"]
    assert len(default) == 0

def test_repeat_aggregate_displays_stats():
    entries = load_fixture("repeat-aggregate.json")
    assert all(e["trend_status"] == "default" for e in entries)
    for e in entries:
        agg = e["canonical_aggregate"]
        assert "method" in agg
        assert "count" in agg
        assert agg["count"] == 3
```

### 3.3 Viewing full coverage

Run the contract tests to see all six categories:

```bash
cd /tmp/vllm-hust-benchmark-issue-79/T13
PYTHONPATH=src python3 -m pytest tests/test_website_contract.py -v
```

This outputs the status for each category and verifies the filtering contract.

---

## 4. data contract invariants

| Invariant | Enforcement | Category |
|-----------|-----------|----------|
| Only `default` entries are formal trends | Website filter | All |
| `experimental` includes a reason | Validator | Experimental |
| `blocked` entries have actionable `trend_reason` | Validator schema | Blocked |
| `invalid` entries are hidden | Website filter | Invalid |
| `excluded` entries are provenance-only | Website filter | Legacy |
| Each entry carries `trend_status` | Schema required | All |
| Aggregate entries declare `canonical_aggregate.count` | Schema conditional check | Repeat aggregate |

---

## 5. Workflow: Adding a new fixture category

1. Create the fixture JSON in `tests/fixtures/trend_coverage/valid/`
2. Add a `CATEGORIES` entry in `tests/test_website_contract.py`
3. Add individual status assertion and website filtering tests
4. Run `pytest tests/test_website_contract.py` to verify
