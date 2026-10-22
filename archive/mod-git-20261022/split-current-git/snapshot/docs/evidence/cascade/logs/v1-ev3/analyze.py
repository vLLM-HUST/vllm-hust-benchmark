"""Final ev3 analysis: build the OFF/ON delta table from saved artifacts only.

Every number printed here is read from a file under this directory (the harness
stdout logs + results/*.json).  Nothing is retyped from memory.
"""
import json
import os
import re
import statistics
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP_RE = re.compile(
    r"SWEEP mode=(\S+) tag=(\S+) (p\d+_b\d+) wall=([\d.]+)s out_tokens=(\d+)"
)

# tag -> human label / leg group
LEGS = {
    "off64_a": "OFF graph", "off64_b": "OFF graph", "off64_c": "OFF graph",
    "ong64_c": "ON graph (shimmed)", "ong64_d": "ON graph (shimmed)",
    "ong64_e": "ON graph (shimmed)",
    "off32_a": "OFF graph", "off32_b": "OFF graph", "off32_c": "OFF graph",
    "ong32_a": "ON graph (shimmed)", "ong32_b": "ON graph (shimmed)",
    "ong32_c": "ON graph (shimmed)",
    "offe64_a": "OFF eager", "one64_a": "ON eager",
    "gate32_v1": "ON graph + W2 gate (shimmed)",
}

runs = []
for log in sorted(os.listdir(HERE)):
    if not log.startswith("sweep_") or not log.endswith(".log"):
        continue
    path = os.path.join(HERE, log)
    with open(path, errors="replace") as fh:
        for line in fh:
            m = SWEEP_RE.search(line)
            if not m:
                continue
            mode, tag, cell, wall, toks = m.groups()
            if tag not in LEGS:
                continue
            runs.append({"leg": LEGS[tag], "mode": mode, "tag": tag, "cell": cell,
                         "wall_s": float(wall), "out_tokens": int(toks),
                         "log": log})

# ---- engagement evidence per ON leg --------------------------------------
def engagement(log):
    path = os.path.join(HERE, log)
    if not os.path.exists(path):
        return {}
    txt = open(path, errors="replace").read()
    marker = re.search(r"cascade plugin loaded \((gate=\d, graph_gate=\d, "
                       r"kernel_wheel=\w+)\)", txt)
    return {
        "marker": marker.group(1) if marker else None,
        "capture_body": len(re.findall(r"capture body: num_tokens=", txt)),
        "replay_cascade_hit": len(re.findall(
            r"replay update: cascade key hit", txt)),
        "replay_gate_off": len(re.findall(r"replay update: gate off", txt)),
        "wrapper_gate_off": len(re.findall(r"wrapper: gate off", txt)),
        "cascade_active_eager": len(re.findall(r"\[cascade-active\]", txt)),
        "twin_miss": len(re.findall(r"twin miss", txt)),
    }

# ---- group by (leg, cell) -------------------------------------------------
groups = {}
for r in runs:
    groups.setdefault((r["leg"], r["cell"]), []).append(r)

summary = {}
for (leg, cell), recs in groups.items():
    walls = sorted(x["wall_s"] for x in recs)
    summary[(leg, cell)] = {
        "n": len(walls), "walls": walls, "median": statistics.median(walls),
        "min": walls[0], "max": walls[-1],
        "spread_pct": (walls[-1] - walls[0]) / walls[0] * 100.0,
        "out_tokens": sorted({x["out_tokens"] for x in recs}),
        "logs": sorted({x["log"] for x in recs}),
    }

def med(leg, cell):
    s = summary.get((leg, cell))
    return s["median"] if s else None

print("=" * 96)
print("PER-RUN RAW (every SWEEP line, from the named log)")
print("=" * 96)
for r in sorted(runs, key=lambda x: (x["cell"], x["leg"], x["wall_s"])):
    print(f"{r['cell']:<11} {r['leg']:<30} wall={r['wall_s']:>6.2f}s "
          f"out={r['out_tokens']:>5}  {r['log']}")

print()
print("=" * 96)
print("DELTA TABLE  (median of repeats; negative = cascade faster)")
print("=" * 96)
hdr = (f"{'cell':<11} {'OFF med':>8} {'ON med':>8} {'delta':>8} "
       f"{'verdict':<6} {'n off/on':>9} {'spread off/on %':>16}")
print(hdr)
print("-" * len(hdr))
delta_rows = []
for cell in sorted({r["cell"] for r in runs}):
    off = med("OFF graph", cell)
    on = med("ON graph (shimmed)", cell)
    if off is None or on is None:
        continue
    d = (on - off) / off * 100.0
    so = summary[("OFF graph", cell)]
    sn = summary[("ON graph (shimmed)", cell)]
    delta_rows.append({"cell": cell, "off_median": off, "on_median": on,
                       "delta_pct": d, "verdict": "WIN" if d < 0 else "LOSS",
                       "n_off": so["n"], "n_on": sn["n"],
                       "off_walls": so["walls"], "on_walls": sn["walls"],
                       "spread_off_pct": so["spread_pct"],
                       "spread_on_pct": sn["spread_pct"],
                       "off_logs": so["logs"], "on_logs": sn["logs"]})
    print(f"{cell:<11} {off:>8.2f} {on:>8.2f} {d:>+7.1f}% "
          f"{'WIN' if d < 0 else 'LOSS':<6} "
          f"{so['n']:>4}/{sn['n']:<4} "
          f"{so['spread_pct']:>7.1f}/{sn['spread_pct']:<8.1f}")

print()
print("EAGER LANE (cascade decode only, no graph capture)")
for cell in sorted({r["cell"] for r in runs}):
    off = med("OFF eager", cell)
    on = med("ON eager", cell)
    if off is None or on is None:
        continue
    d = (on - off) / off * 100.0
    print(f"{cell:<11} OFF {off:>6.2f}s  ON {on:>6.2f}s  {d:>+7.1f}%  "
          f"{'WIN' if d < 0 else 'LOSS'}")

print()
print("W2 GATE LEG (B=32, adaptive gate ON; p420_b32 = measured loss cell)")
for cell in sorted({r["cell"] for r in runs if r["leg"].startswith("ON graph + W2")}):
    off = med("OFF graph", cell)
    on = med("ON graph (shimmed)", cell)
    g = med("ON graph + W2 gate (shimmed)", cell)
    print(f"{cell:<11} OFF {off:>6.2f}s  ON(ungated) {on:>6.2f}s  "
          f"ON(gated) {g:>6.2f}s   gated vs OFF {(g-off)/off*100:+.1f}%")

print()
print("=" * 96)
print("ENGAGEMENT EVIDENCE (per ON log)")
print("=" * 96)
eng = {}
for tag, leg in sorted(LEGS.items()):
    if not leg.startswith("ON"):
        continue
    logs = sorted({r["log"] for r in runs if r["tag"] == tag})
    for lg in logs:
        if lg in eng:
            continue
        eng[lg] = {"tag": tag, "leg": leg, **engagement(lg)}
        e = eng[lg]
        print(f"{lg}\n    marker={e['marker']} capture_body={e['capture_body']} "
              f"replay_cascade_hit={e['replay_cascade_hit']} "
              f"replay_gate_off={e['replay_gate_off']} "
              f"wrapper_gate_off={e['wrapper_gate_off']} "
              f"cascade_active_eager={e['cascade_active_eager']} "
              f"twin_miss={e['twin_miss']}")

# OFF legs must show gate=0 and zero engagement
print()
print("OFF-leg negative controls (must show gate=0, no cascade activity)")
for tag in ("off64_a", "off32_a", "offe64_a"):
    logs = sorted({r["log"] for r in runs if r["tag"] == tag})
    for lg in logs:
        e = engagement(lg)
        print(f"{lg}\n    marker={e['marker']} capture_body={e['capture_body']} "
              f"replay_cascade_hit={e['replay_cascade_hit']} "
              f"cascade_active_eager={e['cascade_active_eager']}")

# token sanity: OFF and ON must both produce full-length outputs
print()
print("output-token sanity (out_tokens per leg/cell)")
for (leg, cell), s in sorted(summary.items()):
    print(f"{cell:<11} {leg:<32} out_tokens={s['out_tokens']}")

out = {"runs": runs, "summary": {f"{k[0]}|{k[1]}": v for k, v in summary.items()},
       "delta_table": delta_rows, "engagement": eng}
with open(os.path.join(HERE, "analysis.json"), "w") as fh:
    json.dump(out, fh, indent=2, sort_keys=True)
print("\nwrote", os.path.join(HERE, "analysis.json"))
