"""Generate a small adversarial SimLLM reuse workload from a pinned warmup fixture.

The edited fact or instruction is near the start of the prompt, well before
the target suffix recomputation. Later variants grow slightly to exercise
approximate source turnover. This generator does not grade model outputs.
"""

import argparse
import hashlib
import json
from pathlib import Path

FIXTURE_SHA256 = "90f3c3229d96237b389e2a3f45fa8f8618d270921091211166661fec4a4e2c41"  # pragma: allowlist secret
SOURCE_COMMIT = "28f2297c58cec2249576dff2ccdd251c662ad8d6"  # pragma: allowlist secret
CASES = (
    ("number", ("17", "29", "43"), "本轮编号是多少？只输出数字。"),
    ("negation", ("允许", "禁止", "允许"), "本轮是否允许操作？只输出‘允许’或‘禁止’。"),
    ("subject", ("张岚", "李诚", "王宁"), "本轮负责人是谁？只输出人名。"),
    ("instruction", ("A", "B", "C"), "执行本轮指令，只输出指定的大写字母。"),
)


def make(fixture: Path) -> list[dict]:
    raw = fixture.read_bytes()
    if hashlib.sha256(raw).hexdigest() != FIXTURE_SHA256:
        raise ValueError("Warmup fixture differs from the pinned source")
    source = [json.loads(line) for line in raw.splitlines() if line]
    groups = []
    seen = set()
    for row in source:
        if row["group_id"] not in seen:
            groups.append(row)
            seen.add(row["group_id"])
    if len(groups) < len(CASES):
        raise ValueError("Not enough source groups")

    output = []
    for (kind, values, question), base in zip(CASES, groups, strict=False):
        background = base["messages"][0]["content"].split("【任务】", 1)[0]
        if "【任务】" not in base["messages"][0]["content"]:
            raise ValueError("Missing fixture task delimiter")
        cut = background.find("\n\n", 1200)
        if cut < 0 or len(background) - cut < 1000:
            raise ValueError("Fixture has no safe interior split")
        prefix, suffix = background[:cut], background[cut:]
        for variant, value in enumerate(values):
            if kind == "number":
                key = f"【本轮唯一有效记录】本轮编号：{value}。其他数字均属于背景材料。"
            elif kind == "negation":
                key = f"【本轮唯一有效记录】本轮是否允许操作：{value}。其他规则均属于背景材料。"
            elif kind == "subject":
                key = (
                    f"【本轮唯一有效记录】本轮负责人：{value}。其他人名均属于背景材料。"
                )
            else:
                key = f"【本轮指令】只输出大写字母 {value}，不得输出其他字母。"
            growth = "补充说明：该段仅提供背景，不覆盖本轮记录或指令。" * (variant * 2)
            prompt = (
                "请依据本轮唯一有效记录或指令作答。\n\n"
                + prefix
                + "\n\n"
                + key
                + "\n\n"
                + suffix
                + "\n"
                + growth
                + "\n【本轮问题】"
                + question
                + "不要解释，不要输出额外文字。"
            )
            output.append(
                {
                    "id": f"issue3_{kind}_{variant}",
                    "group_id": f"issue3_{kind}",
                    "topic": base["topic"],
                    "prompt": prompt,
                    "messages": [{"role": "user", "content": prompt}],
                    "expected": value,
                    "change_kind": kind,
                    "variant": variant,
                    "source_fixture_group": base["group_id"],
                    "source_commit": SOURCE_COMMIT,
                    "metadata": {
                        "reuse_tier": "high",
                        "matched_flag": variant > 0,
                        "output_tokens_target": 64,
                    },
                }
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = make(args.fixture)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )


if __name__ == "__main__":
    main()
