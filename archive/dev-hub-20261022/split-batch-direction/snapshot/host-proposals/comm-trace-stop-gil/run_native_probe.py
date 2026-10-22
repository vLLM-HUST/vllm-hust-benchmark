# SPDX-License-Identifier: Apache-2.0
import argparse
import importlib.util
import json
import subprocess
import sysconfig
from pathlib import Path

from gen_patch import ORIGINAL, patched_source

PREAMBLE = r"""
#include <Python.h>
#include <pybind11/pybind11.h>
#include <atomic>
#include <deque>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#define TORCH_CHECK(condition, ...) \
    do { if (!(condition)) throw std::runtime_error("active tracer"); } while (0)
"""
STATE = r"""
class PythonTracer {
 public:
  void clear();
  std::atomic<bool> active_{false};
  std::vector<int> py_call_cache_, pyc_call_cache_, module_info_cache_;
  std::vector<int> module_param_cache_, optimizer_param_cache_, events_;
  std::vector<int> ctx_tid_map_, start_py_call_info_;
  std::deque<ThreadLocalResult> thread_local_results_;
  PyInterpreterState* interpreter_{nullptr};
};
"""
MAIN = r"""
int main(int argc, char** argv) {
  Py_Initialize();
  if (PyType_Ready(&TraceContextType) != 0) return 2;
  PythonTracer tracer;
  for (int i = 0; i < 128; ++i) tracer.thread_local_results_.emplace_back(&tracer);
  bool release = argc > 1 && std::string(argv[1]) == "released";
  PyThreadState* state = release ? PyEval_SaveThread() : nullptr;
  tracer.clear();
  if (release) PyEval_RestoreThread(state);
  if (!tracer.thread_local_results_.empty()) return 3;
  std::cout << "clear completed; GIL=" << PyGILState_Check() << std::endl;
  Py_Finalize();
  return 0;
}
"""


def native_source(source):
    start = source.index("struct ThreadLocalResult;")
    end = source.index("\nstruct PyCallInfo", start)
    clear_start = source.index("void PythonTracer::clear() {")
    clear_end = source.index("\n}\n", clear_start) + 3
    return PREAMBLE + source[start:end] + STATE + source[clear_start:clear_end] + MAIN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cxx", default="c++")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is None:
        raise SystemExit("torch headers are required; no device initialization is used")
    torch_include = Path(torch_spec.origin).parent / "include"
    include = sysconfig.get_path("include")
    libdir = sysconfig.get_config_var("LIBDIR")
    library = sysconfig.get_config_var("LDLIBRARY")
    linker_name = library.removeprefix("lib").split(".so")[0]
    original = ORIGINAL.read_bytes()
    sources = {"original": original.decode(), "fixed": patched_source(original)}
    results = []
    for name, text in sources.items():
        src = args.output / f"{name}.cpp"
        binary = args.output / name
        src.write_text(native_source(text))
        command = [
            args.cxx,
            "-std=c++17",
            "-O0",
            "-g",
            str(src),
            f"-I{include}",
            f"-I{torch_include}",
            f"-L{libdir}",
            f"-Wl,-rpath,{libdir}",
            f"-l{linker_name}",
            "-o",
            str(binary),
        ]
        build = subprocess.run(command, capture_output=True, text=True, timeout=90)
        (args.output / f"{name}-build.log").write_text(build.stdout + build.stderr)
        if build.returncode:
            raise SystemExit(f"{name} build failed: see {args.output}")
        for mode in ["held", "released"]:
            for repeat in range(3):
                run = subprocess.run(
                    [str(binary), mode], capture_output=True, text=True, timeout=30
                )
                (args.output / f"{name}-{mode}-{repeat}.log").write_text(
                    run.stdout + run.stderr
                )
                results.append(
                    {
                        "source": name,
                        "gil": mode,
                        "repeat": repeat,
                        "returncode": run.returncode,
                    }
                )
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    for result in results:
        expected = (
            -11 if (result["source"], result["gil"]) == ("original", "released") else 0
        )
        if result["returncode"] != expected:
            raise SystemExit("native clear regression did not match expected behavior")


if __name__ == "__main__":
    main()
