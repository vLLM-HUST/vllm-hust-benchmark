"""Software-only AST fixtures, not native model execution or perplexity evidence."""

import ast
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from temporalwiki_original_binding import baseline_factory_tree, npu_evaluator_tree


FACTORY = """
import pytorch_lightning as pl
from deepspeed.runtime.lr_schedules import WarmupDecayLR
import deepspeed
from models.GPT2_Model_Kadapter import GPT2LMHeadModel as GPT2_Kadapter
from models.GPT2_Model_LoRA import GPT2LMHeadModel as GPT2_Lora
from models.RecAdam import RecAdam
class GPT2(pl.LightningModule):
    def forward(self, x):
        return self.model(x)
"""


def evaluator():
    calls = "\n".join(f"    x{i} = tensor.cuda()" for i in range(14))
    return f"""
def evaluate_ppl(args, Model):
    model = Model(args)
    model.to('cuda')
{calls}
    total_loss += loss
    batch_cnt += 1
    return torch.exp(total_loss / batch_cnt)
def evaluate_ppl_corpus(args, Model):
    return tensor.cuda()
"""


def test_factory_keeps_original_class_and_real_dependency():
    before = ast.parse(FACTORY)
    tree, removed = baseline_factory_tree(FACTORY)
    assert len(removed) == 5
    assert ast.dump(tree.body[-1]) == ast.dump(before.body[-1])
    assert ast.unparse(tree.body[0]) == "import pytorch_lightning as pl"


def test_factory_rejects_missing_or_mixed_training_imports():
    with pytest.raises(ValueError):
        baseline_factory_tree(FACTORY.replace("import deepspeed\n", ""))
    with pytest.raises(ValueError):
        baseline_factory_tree(
            FACTORY.replace("import deepspeed\n", "import deepspeed, os\n")
        )


def test_only_placements_change_not_loss_or_other_protocol():
    original = ast.parse(evaluator())
    tree, counts = npu_evaluator_tree(evaluator(), "npu:1")
    assert counts == {"tensor_cuda_to_npu": 14, "model_cuda_to_npu": 1}
    assert ast.dump(original.body[1]) == ast.dump(tree.body[1])
    assert ast.dump(original.body[0].body[-1]) == ast.dump(tree.body[0].body[-1])
    assert ast.unparse(tree.body[0]).count(".to('npu:1')") == 15


@pytest.mark.parametrize("device", ["cpu", "cuda:0", "npu", "npu:-1"])
def test_no_other_device_or_silent_fallback(device):
    with pytest.raises(ValueError):
        npu_evaluator_tree(evaluator(), device)


def test_changed_official_placement_contract_is_rejected():
    with pytest.raises(ValueError):
        npu_evaluator_tree(evaluator().replace("x6 = tensor.cuda()", "x6 = tensor"))
