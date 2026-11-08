"""Explicit baseline-only compatibility transforms; no model or scorer replacement.

Callers must verify the pinned author source before using these transforms.
The returned ASTs retain original model bodies and perplexity aggregation.
"""

import ast
import re


TRAINING_IMPORTS = {
    "deepspeed.runtime.lr_schedules",
    "models.GPT2_Model_Kadapter",
    "models.GPT2_Model_LoRA",
    "models.RecAdam",
}


def baseline_factory_tree(raw):
    """Exclude five unused training imports, without changing original classes."""
    tree = ast.parse(raw)
    retained, removed = [], []
    for node in tree.body:
        if isinstance(node, ast.Import) and any(
            item.name == "deepspeed" for item in node.names
        ):
            if len(node.names) != 1:
                raise ValueError("unexpected mixed deepspeed import")
            removed.append("deepspeed")
        elif isinstance(node, ast.ImportFrom) and node.module in TRAINING_IMPORTS:
            removed.append(node.module)
        else:
            retained.append(node)
    if set(removed) != TRAINING_IMPORTS | {"deepspeed"} or len(removed) != 5:
        raise ValueError("original five training-only import contract changed")
    tree.body = retained
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    if len(classes) != 1 or classes[0].name != "GPT2":
        raise ValueError("original GPT2 class contract changed")
    return tree, removed


def npu_evaluator_tree(raw, device="npu:0"):
    """Bind evaluate_ppl's 15 placements, including its inactive corpus branch."""
    if not re.fullmatch(r"npu:[0-9]+", device):
        raise ValueError("explicit NPU device required; CPU/CUDA fallback forbidden")
    tree = ast.parse(raw)
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "evaluate_ppl"
    ]
    if len(functions) != 1:
        raise ValueError("one original evaluate_ppl function required")

    class Placement(ast.NodeTransformer):
        def __init__(self):
            self.cuda_calls = 0
            self.model_to = 0

        def visit_Call(self, node):
            self.generic_visit(node)
            if not isinstance(node.func, ast.Attribute):
                return node
            if node.func.attr == "cuda":
                if node.args or node.keywords:
                    raise ValueError("unexpected original cuda arguments")
                node.func.attr = "to"
                node.args = [ast.Constant(device)]
                self.cuda_calls += 1
            elif (
                node.func.attr == "to"
                and len(node.args) == 1
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "cuda"
            ):
                if node.keywords:
                    raise ValueError("unexpected original to arguments")
                node.args[0] = ast.Constant(device)
                self.model_to += 1
            return node

    transform = Placement()
    transform.visit(functions[0])
    if (transform.cuda_calls, transform.model_to) != (14, 1):
        raise ValueError(
            "original evaluate_ppl CUDA placement contract changed: "
            f"tensor={transform.cuda_calls}, model={transform.model_to}"
        )
    ast.fix_missing_locations(tree)
    return tree, {"tensor_cuda_to_npu": 14, "model_cuda_to_npu": 1}
