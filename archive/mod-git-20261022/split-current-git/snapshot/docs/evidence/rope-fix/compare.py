#!/usr/bin/env python3
"""Build the RoPE-fix evidence comparison table and check the claims.

Compares four probe artefacts of the SAME probe script
(``knowledge/surveys/contrast/norm-rope-act/rope_variants_e2e_probe.py``,
copied verbatim into /tmp for the runs so the read-only knowledge tree is never
overwritten):

  prefix      knowledge/.../logs_prefix_2026-09-10/rope_variants_e2e.json
              (fork baseline, no fix anywhere)
  control-off logs/control-off-main.json
              (this carrier NOT enabled: VLLM_HUST_ROPE_FIX unset)
  carrier-on  logs/carrier-main.json + logs/carrier-atb.json
              (this carrier enabled: VLLM_HUST_ROPE_FIX=1, no PYTHONPATH override)
  forkfix     reference/forkfix-main.json
              (fork branch /tmp/vah-ropefix through PYTHONPATH, the previous form)

Judgement: the carrier legs must reproduce the forkfix values (fix landed) and the
control-off leg must reproduce the prefix values (default-off is bit-identical /
the three breakpoints still reproduce when the carrier is off).

Usage: python compare.py [--check]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VARIANTS = ("interleaved", "llama3", "yarn", "mrope")

SOURCES = {
    "prefix": HERE / "reference" / "prefix-main.json",
    "control-off": HERE / "logs" / "control-off-main.json",
    "carrier-on": HERE / "logs" / "carrier-main.json",
    "forkfix": HERE / "reference" / "forkfix-main.json",
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _row(data: dict, variant: str) -> dict:
    cases = [case for case in data["cases"] if case["variant"] == variant]
    assert cases, variant
    first = cases[0]
    if variant == "mrope":
        # mrope has no ``prod_forward`` leg at all (the probe exercises the
        # wrapper instead), so an ``all(...)`` over the remaining cases would be
        # vacuously True.  Report the actual criterion for this variant.
        wiring = (
            "True"
            if all(c.get("kernel_wrapper", {}).get("ok") for c in cases)
            else "**False**"
        )
    else:
        wiring = all(
            bool(
                case.get("prod_forward", {})
                .get("o_vs_direct_kernel", {})
                .get("bitwise")
            )
            for case in cases
        )
    return {
        "cls": first["gen"]["cls"],
        "fwd": first["gen"]["fwd_method"],
        "reachable": data["summary"][variant]["kernel_reachable"],
        "e2e_status": data["summary"][variant]["e2e"]["status"],
        "chain_selfconsistent_max_abs": data["summary"][variant]["e2e"][
            "chain_selfconsistent_max_abs"
        ],
        "vs_host_formula_max_abs": data["summary"][variant]["e2e"][
            "vs_host_formula_max_abs"
        ],
        "formula_divergence_max": data["summary"][variant]["e2e"][
            "formula_divergence_max"
        ],
        # non-mrope: production forward ≡ the direct kernel call (O≡直接kernel);
        # mrope:    the wrapper call returns instead of raising (包装正常返回).
        "wiring_proof": wiring,
        "wiring_label": "包装正常返回" if variant == "mrope" else "O≡直接kernel",
        "mrope_wrapper_ok": (
            all(bool(case.get("kernel_wrapper", {}).get("ok")) for case in cases)
            if variant == "mrope"
            else None
        ),
        "n_bad_pre": [case["e2e"]["n_bad_pre"] for case in cases],
    }


def _fmt(value) -> str:
    if isinstance(value, bool):
        return "True" if value else "**False**"
    if isinstance(value, float):
        return f"{value:.6g}"
    if value is None:
        return "—"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="assert the claims")
    args = parser.parse_args()

    data = {name: _load(path) for name, path in SOURCES.items()}
    rows = {
        name: {v: _row(payload, v) for v in VARIANTS} for name, payload in data.items()
    }

    print("## 探针四腿对比（四变体 × 四腿）\n")
    header = (
        "| 腿 | 生产类 | 实际 fwd | kernel_reachable | 接线判据 "
        "| e2e 判定 | 对 host 公式偏差 | 公式分歧 |"
    )
    for variant in VARIANTS:
        print(f"### {variant}\n")
        print(header)
        print("|---|---|---|---|---|---|---|---|")
        for name in ("prefix", "control-off", "carrier-on", "forkfix"):
            row = rows[name][variant]
            print(
                f"| {name} | `{row['cls']}` | `{row['fwd'].split('.')[-1]}` | "
                f"{row['reachable']} | {_fmt(row['wiring_proof'])} "
                f"({row['wiring_label']}) | "
                f"{row['e2e_status']} | {_fmt(row['vs_host_formula_max_abs'])} | "
                f"{_fmt(row['formula_divergence_max'])} |"
            )
        print()

    print("## 判定字段（carrier-on vs forkfix / control-off vs prefix）\n")
    fields = (
        "cls",
        "reachable",
        "e2e_status",
        "chain_selfconsistent_max_abs",
        "vs_host_formula_max_abs",
        "formula_divergence_max",
        "wiring_proof",
        "mrope_wrapper_ok",
    )
    verdicts = {"carrier_vs_forkfix": True, "control_off_vs_prefix": True}
    for variant in VARIANTS:
        for left, right, key in (
            ("carrier-on", "forkfix", "carrier_vs_forkfix"),
            ("control-off", "prefix", "control_off_vs_prefix"),
        ):
            for field in fields:
                a, b = rows[left][variant][field], rows[right][variant][field]
                same = a == b
                # The plugin subclasses are *different classes by construction*;
                # what must match is the wiring fact, not the name.
                if field == "cls" and left == "carrier-on":
                    same = b.startswith("Ascend") and a.startswith("AscendFix")
                    if variant == "interleaved":
                        same = a == b
                if not same:
                    verdicts[key] = False
                    print(
                        f"- MISMATCH {key}: {variant}.{field} "
                        f"carrier={a!r} reference={b!r}"
                    )
    for key, ok in verdicts.items():
        print(f"- {key}: {'PASS' if ok else 'FAIL'}")

    print("\n## 附加：OFF 腿复现修前三断点\n")
    off = data["control-off"]
    off_mrope = next(case for case in off["cases"] if case["variant"] == "mrope")
    off_llama3 = rows["control-off"]["llama3"]
    off_yarn = rows["control-off"]["yarn"]
    default_note = "（= host 类 + `CustomOp.forward_oot` 默认实现）"
    print(
        f"- llama3 生产类 `{off_llama3['cls']}` / fwd "
        f"`{off_llama3['fwd']}`{default_note}"
    )
    print(
        f"- mrope 包装调用：`{off_mrope['kernel_wrapper'].get('error')}`"
        f"（位置 {off_mrope['kernel_wrapper'].get('where')}）"
    )
    print(
        f"- yarn 公式分歧：{_fmt(off_yarn['formula_divergence_max'])}"
        f"（端到端偏差 {_fmt(off_yarn['vs_host_formula_max_abs'])}）"
    )
    print(f"- control-off JSON `oot_rope_keys` = {off['meta']['oot_rope_keys']}")
    print(
        "- carrier-on JSON `oot_rope_keys` = "
        f"{data['carrier-on']['meta']['oot_rope_keys']}"
    )

    if args.check:
        return 0 if all(verdicts.values()) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
