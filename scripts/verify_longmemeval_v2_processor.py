"""Verify original reader image/token preprocessing, never execute a neural model."""

import argparse
import json
import os
from pathlib import Path
import subprocess

from prepare_xsum_original_assets import verify

PINS = (
    Path(__file__).resolve().parents[1]
    / "configs/workload-native/longmemeval-v2-reader-pins.json"
)
PROBE = r"""
import json,sys,torch
from PIL import Image
from transformers import AutoProcessor,PreTrainedModel
def forbidden(*args,**kwargs):
    raise AssertionError('neural forward or weight loading forbidden in CPU processor probe')
torch.nn.Module._call_impl=forbidden
PreTrainedModel.from_pretrained=forbidden
torch.load=forbidden
assert torch.version.cuda is None and 'torch_npu' not in sys.modules
processor=AutoProcessor.from_pretrained(sys.argv[1],local_files_only=True)
image=Image.new('RGB',(224,224))
messages=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':'inert processor fixture'}]}]
text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
result=processor(text=[text],images=[image],return_tensors='pt')
assert result['input_ids'].device.type=='cpu' and result['pixel_values'].device.type=='cpu'
print(json.dumps({'status':'ORIGINAL_READER_PROCESSOR_FIXTURE_PASSED_NOT_NATIVE_RAG','torch':torch.__version__,'cuda_build':torch.version.cuda,'processor_class':type(processor).__name__,'input_tokens':result['input_ids'].shape[-1],'pixel_tensor_shape':list(result['pixel_values'].shape),'neural_module_calls':0,'model_weight_loads':0,'new_model_calls':0,'source_checkbox_eligible':False}))
"""


def verify_metadata(root):
    root = Path(root).resolve(strict=True)
    pins = json.loads(PINS.read_text("utf-8"))
    metadata = [
        item
        for item in pins["model"]["files"]
        if not item["path"].endswith(".safetensors")
    ]
    if not metadata:
        raise ValueError("fixed original processor metadata required")
    for item in metadata:
        path = (root / item["path"]).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError("processor asset escaped original root")
        verify(path, item)
    return len(metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--reader-assets", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "PLAN_ONLY",
                    "new_model_calls": 0,
                    "source_checkbox_eligible": False,
                }
            )
        )
        return
    before = verify_metadata(args.reader_assets)
    interpreter = args.python.absolute()
    if not interpreter.is_file():
        raise FileNotFoundError("existing isolated client Python required")
    environment = dict(
        os.environ,
        PYTHONNOUSERSITE="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
        CUDA_VISIBLE_DEVICES="",
        ASCEND_RT_VISIBLE_DEVICES="",
    )
    for name in ("PYTHONPATH", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "OPENAI_API_KEY"):
        environment.pop(name, None)
    result = subprocess.run(
        [
            str(interpreter),
            "-IB",
            "-c",
            PROBE,
            str(args.reader_assets.resolve(strict=True)),
        ],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(result.stdout)
    if (
        report.get("status")
        != "ORIGINAL_READER_PROCESSOR_FIXTURE_PASSED_NOT_NATIVE_RAG"
        or report.get("new_model_calls") != 0
        or report.get("source_checkbox_eligible") is not False
    ):
        raise ValueError("unexpected processor fixture receipt")
    report["original_metadata_files_verified_before_and_after"] = before
    if verify_metadata(args.reader_assets) != before:
        raise ValueError("processor metadata set changed")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
