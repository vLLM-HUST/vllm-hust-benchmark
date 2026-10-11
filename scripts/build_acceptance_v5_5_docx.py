#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import NamedTemporaryFile
from xml.etree import ElementTree
from zipfile import ZIP_DEFLATED, ZipFile


OLD_OWNER = "B0执行工程师"
NEW_OWNER = "课题组（发布与评测工程）"
OLD_ACTION = (
    "提供与该结果表SHA-256相互绑定的完整server argv、client contract、"
    "effective/startup receipt、模型与tokenizer manifest、数据及到达序列、镜像和代码身份"
)
NEW_ACTION = (
    "按表附-1至表附-9生成并冻结签名release manifest、两侧OCI RepoDigest、完整server argv/"
    "client contract和effective/startup receipt；与模型、tokenizer、数据、到达序列、结果表及"
    "原始请求建立校验绑定。"
)
OLD_RELEASE = (
    "所有性能相关字段可逐项生成唯一命令，并与结果表及原始请求统一签名后，才可纳入正式B0"
)
NEW_RELEASE = (
    "发布工件与实际运行生成的材料全部纳入同一不可变manifest并验签通过；字段可生成唯一命令且"
    "与生效值一致，方可纳入正式B0。"
)
FORBIDDEN_TEXT = (
    "课题组自行决定",
    "无需甲方确认",
    "无需浦江确认",
    "无需第三方确认",
    "封版权限",
)


def replace_once(data: bytes, old: str, new: str, *, label: str) -> bytes:
    old_bytes = old.encode("utf-8")
    count = data.count(old_bytes)
    if count != 1:
        raise ValueError(f"expected one {label} occurrence, found {count}")
    return data.replace(old_bytes, new.encode("utf-8"))


def transform_entry(name: str, data: bytes) -> bytes:
    if name == "word/document.xml":
        data = data.replace(b"V5.4", b"V5.5")
        data = replace_once(
            data, "2026年10月10日", "2026年10月11日", label="cover date"
        )
        data = replace_once(data, OLD_OWNER, NEW_OWNER, label="B0 receipt owner")
        data = replace_once(data, OLD_ACTION, NEW_ACTION, label="B0 receipt action")
        data = replace_once(
            data, OLD_RELEASE, NEW_RELEASE, label="B0 receipt release criterion"
        )
    elif name == "docProps/core.xml":
        data = data.replace(b"V5.4", b"V5.5")
        data = data.replace(b"2026-10-09T08:07:32Z", b"2026-10-11T00:00:00Z")
        data = data.replace(
            b"<cp:revision>54</cp:revision>", b"<cp:revision>55</cp:revision>"
        )
    return data


def document_text(path: Path) -> str:
    with ZipFile(path) as archive:
        root = ElementTree.fromstring(archive.read("word/document.xml"))
    return "".join(root.itertext())


def validate_output(path: Path) -> None:
    text = document_text(path)
    required = (
        "V5.5",
        "2026年10月11日",
        NEW_OWNER,
        NEW_ACTION,
        NEW_RELEASE,
    )
    for value in required:
        if value not in text:
            raise ValueError(f"generated document is missing required text: {value}")
    for value in ("V5.4", OLD_OWNER, OLD_ACTION, OLD_RELEASE, *FORBIDDEN_TEXT):
        if value in text:
            raise ValueError(
                f"generated document contains stale or forbidden text: {value}"
            )


def build(source: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=output.parent, suffix=".docx", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with (
            ZipFile(source) as source_archive,
            ZipFile(temporary_path, "w", compression=ZIP_DEFLATED) as output_archive,
        ):
            for info in source_archive.infolist():
                output_archive.writestr(
                    info,
                    transform_entry(info.filename, source_archive.read(info.filename)),
                )
        validate_output(temporary_path)
        temporary_path.replace(output)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the V5.5 DOCX semantic mirror")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.source, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
