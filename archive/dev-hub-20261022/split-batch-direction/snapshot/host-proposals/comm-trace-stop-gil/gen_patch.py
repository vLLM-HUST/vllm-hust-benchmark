# SPDX-License-Identifier: Apache-2.0
import argparse
import difflib
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
HOST_REL = "torch_npu/csrc/profiler/profiler_python.cpp"
ORIGINAL = HERE / "host_orig/profiler_python.cpp.orig"
EXPECTED_SHA256 = "8469611bd1d08bcd12201058089965d31bcd0df490d046f547f7b7e364066a55"
ANCHOR = "void PythonTracer::clear() {\n"


def patched_source(source):
    if hashlib.sha256(source).hexdigest() != EXPECTED_SHA256:
        raise ValueError("profiler_python.cpp differs from pinned torch_npu source")
    text = source.decode()
    if text.count(ANCHOR) != 1:
        raise ValueError("expected exactly one PythonTracer::clear")
    return text.replace(ANCHOR, ANCHOR + "  pybind11::gil_scoped_acquire gil;\n")


def generate(source):
    return "".join(
        difflib.unified_diff(
            source.decode().splitlines(keepends=True),
            patched_source(source).splitlines(keepends=True),
            fromfile=f"a/{HOST_REL}",
            tofile=f"b/{HOST_REL}",
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    source = ORIGINAL.read_bytes()
    expected = generate(source)
    if args.host is not None:
        patched_source(args.host.read_bytes())
    if args.check:
        if (HERE / "patch.diff").read_text() != expected:
            raise SystemExit("patch.diff is stale")
    else:
        print(expected, end="")


if __name__ == "__main__":
    main()
