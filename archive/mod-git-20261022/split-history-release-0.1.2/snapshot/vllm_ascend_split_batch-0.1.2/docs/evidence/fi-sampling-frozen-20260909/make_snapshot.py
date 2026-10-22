#!/usr/bin/env python3
"""重建 W2 `fi_sampling` **源包冻结快照**（源侧 4 个文件的字节）。

为什么需要它（2026-09-26 核实的两条事实）：

1. W2 源包是**活工作区**（`knowledge/surveys/sampling/fi_sampling`），2026-09-09 之后
   又加了 W3 renorm/mask 家族 ⇒ 今天的源码 `diff` 不再是"搬运差异"
   （`api.py` 现 `4e3ce156…`、`pure.py` 现 `8a8bba61…`）。
2. 工作区仓的 git 历史里**没有** W2 那一版：`api.py` 在该仓只有一个版本
   （`9e98559`，2026-09-12 提交，内容即漂移后的 `4e3ce156…`）⇒ **打 tag 冻结不可行**。

唯一可用的重建路径是 `PROVENANCE.md` §4.1 已证实的方法：搬运只改了两条导入行，
其余字节与源侧相同 ⇒ 把 vendored 的相对导入还原为绝对导入，即得源侧字节，
其 sha256 必须命中 §4.1 的记录值 `902884d6…`（该命中同时证明记录为真、且搬运无夹带）。

产物（本目录）：

- `snapshot/{api,kernels,npu_env,pure}.py` —— 源侧字节，第三方可直接 `sha256sum` 比对，
  不再需要自己推逆变换；
- `FROZEN.sha256` —— 上述四件的 sha256，格式与 `sha256sum -c` 兼容。

用法：

    python3 make_snapshot.py           # 重建并写 snapshot/ + FROZEN.sha256
    python3 make_snapshot.py --check   # 只校验入仓件与记录值一致（不写）
    sha256sum -c FROZEN.sha256         # 第三方复核（在本目录内执行）
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
VENDORED = REPO_ROOT / "src" / "vllm_ascend_split_batch" / "fi_sampling"
SNAPSHOT = HERE / "snapshot"
MANIFEST = HERE / "FROZEN.sha256"

#: W2 快照的源侧 sha256（`PROVENANCE.md` §4.1 记录值，逐条可复核）。
SOURCE_SHA256 = {
    "api.py": "902884d6cd64c78b8b917f6404f51305a1f23a37760b80a005466b770918a03c",
    "kernels.py": "8d0b20cca2559a2c4d893a96bc6c2398569eb2805258848f54ccde114b17d27e",
    "npu_env.py": "04a3a122f66a74b7176cfaf7f3b1b701871bfe0e55e252fcf58f8a85b352a77a",
    "pure.py": "923d7db26a704e778dfafb4bd74a2742364e6a5b8de250518bc08dfd6954ab08",
}

#: vendoring 时的导入改写（源码形态 → vendored 形态）；逆变换即据此还原。
#: 与 `tests/test_provenance_hashes.py::VENDORED_IMPORT_REWRITES` 必须一致。
VENDORED_IMPORT_REWRITES = (
    ("from fi_sampling.kernels import", "from .kernels import"),
    ("from fi_sampling.npu_env import", "from .npu_env import"),
)


def build_snapshot() -> dict[str, bytes]:
    """从 vendored 字节重建源侧 4 个文件的字节。"""
    rebuilt: dict[str, bytes] = {}
    for name in SOURCE_SHA256:
        path = VENDORED / name
        if not path.is_file():
            raise SystemExit(f"vendored file missing: {path}")
        text = path.read_text(encoding="utf-8")
        for source_form, vendored_form in VENDORED_IMPORT_REWRITES:
            if vendored_form in text:
                text = text.replace(vendored_form, source_form)
        rebuilt[name] = text.encode()
    return rebuilt


def verify(rebuilt: dict[str, bytes]) -> None:
    """重建件必须命中 §4.1 的记录值；不命中即拒绝写/报错。"""
    problems = []
    for name, expected in SOURCE_SHA256.items():
        actual = hashlib.sha256(rebuilt[name]).hexdigest()
        if actual != expected:
            problems.append(f"{name}: expected {expected}, got {actual}")
    if problems:
        raise SystemExit(
            "rebuilt snapshot does not match the recorded W2 source hashes "
            "(vendored bytes drifted, or the rewrite table is stale):\n  "
            + "\n  ".join(problems)
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed snapshot against the recorded hashes; write nothing",
    )
    args = parser.parse_args(argv)

    rebuilt = build_snapshot()
    verify(rebuilt)

    if args.check:
        committed = {
            name: hashlib.sha256((SNAPSHOT / name).read_bytes()).hexdigest()
            for name in SOURCE_SHA256
        }
        mismatched = {
            name: (SOURCE_SHA256[name], committed[name])
            for name in SOURCE_SHA256
            if committed[name] != SOURCE_SHA256[name]
        }
        if mismatched:
            for name, (expected, actual) in sorted(mismatched.items()):
                print(f"MISMATCH {name}: recorded {expected}, committed {actual}")
            return 1
        print(f"snapshot OK: {len(SOURCE_SHA256)} files match PROVENANCE.md §4.1")
        return 0

    SNAPSHOT.mkdir(parents=True, exist_ok=True)
    lines = []
    for name in sorted(rebuilt):
        (SNAPSHOT / name).write_bytes(rebuilt[name])
        lines.append(f"{hashlib.sha256(rebuilt[name]).hexdigest()}  snapshot/{name}")
    MANIFEST.write_text(
        "# W2 fi_sampling 源侧冻结快照（frozen-20260909）的 sha256。\n"
        "# 重建方式与判据见同目录 README.md；复核：cd 本目录 && sha256sum -c FROZEN.sha256\n"
        + "\n".join(lines)
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(rebuilt)} files to {SNAPSHOT} and {MANIFEST.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
