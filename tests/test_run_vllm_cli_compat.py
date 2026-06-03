from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_vllm_cli_compat.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_vllm_cli_compat", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _register_vllm_modules(*modules: ModuleType) -> None:
    for module in modules:
        sys.modules[module.__name__] = module


def test_load_flexible_argument_parser_prefers_argparse_utils(monkeypatch) -> None:
    module = _load_module()

    vllm_module = ModuleType("vllm")
    vllm_module.__path__ = []

    utils_module = ModuleType("vllm.utils")
    utils_module.__path__ = []

    argparse_utils_module = ModuleType("vllm.utils.argparse_utils")

    class LegacyParser:
        pass

    class ModernParser:
        pass

    utils_module.FlexibleArgumentParser = LegacyParser
    argparse_utils_module.FlexibleArgumentParser = ModernParser

    _register_vllm_modules(vllm_module, utils_module, argparse_utils_module)

    monkeypatch.setattr(module.sys, "modules", sys.modules)

    assert module._load_flexible_argument_parser() is ModernParser


def test_load_flexible_argument_parser_falls_back_to_legacy_utils(monkeypatch) -> None:
    module = _load_module()

    vllm_module = ModuleType("vllm")
    vllm_module.__path__ = []

    utils_module = ModuleType("vllm.utils")
    utils_module.__path__ = []

    class LegacyParser:
        pass

    utils_module.FlexibleArgumentParser = LegacyParser

    _register_vllm_modules(vllm_module, utils_module)
    sys.modules.pop("vllm.utils.argparse_utils", None)

    monkeypatch.setattr(module.sys, "modules", sys.modules)

    assert module._load_flexible_argument_parser() is LegacyParser