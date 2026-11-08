"""Device-only binding for the pinned original Silero TorchScript model."""

import hashlib
from pathlib import Path

MODEL_SHA256 = "e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720"  # pragma: allowlist secret - public model hash


class NpuSilero:
    def __init__(self):
        import torch_npu  # noqa: F401 - registers the NPU backend
        import silero_vad

        model_path = Path(silero_vad.__file__).parent / "data/silero_vad.jit"
        if hashlib.sha256(model_path.read_bytes()).hexdigest() != MODEL_SHA256:
            raise ValueError("pinned original Silero TorchScript model differs")
        self.model = silero_vad.load_silero_vad(onnx=False).to("npu:0").eval()
        if {str(p.device) for p in self.model.parameters()} != {"npu:0"}:
            raise ValueError("all original Silero parameters must be on npu:0")

    def reset_states(self):
        return self.model.reset_states()

    def __call__(self, waveform, sample_rate):
        output = self.model(waveform.to("npu:0"), sample_rate)
        if str(output.device) != "npu:0":
            raise ValueError("Silero output must remain on NPU")
        return output


def load_npu_silero():
    return NpuSilero()
