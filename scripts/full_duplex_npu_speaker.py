"""Explicit NPU binding for the qualified fixed reference-voice encoder.

Configuration selects a relocatable artifact bundle. Only the original fixed
reference features are accepted; this is not a general ONNX CPU fallback.
"""

import atexit
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ORIGINAL_ONNX_SHA256 = "a6ac6a63997761ae2997373e2ee1c47040854b4b759ea41ec48e4e42df0f4d73"  # pragma: allowlist secret - publisher model
DERIVED_GRAPH_SHA256 = "fcaee2adf98e69b1ea1dc3b3b3905c10c097e1a1216d2eb09afa11cd8afb4939"  # pragma: allowlist secret - disclosed lowered graph
INPUT_SHA256 = "17048a4b2ac90f500e36144387bae9382aeb085eec3b1008d5b032a6109c0c56"  # pragma: allowlist secret - fixed reference features
OUTPUT_SHA256 = "42bcfda6172545ee56834b16f8085fdef0e6f1656ea8d297a2b2a2ee6447adaa"  # pragma: allowlist secret - qualified NPU reference output


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4194304), b""):
            result.update(block)
    return result.hexdigest()


def load_profile(path):
    path = Path(path).resolve(strict=True)
    profile = json.loads(path.read_text("utf-8"))
    if profile.get("schema_version") != "full-duplex-npu-speaker-profile/v1":
        raise ValueError("explicit fixed-reference NPU speaker profile required")
    files = {}
    for name in ("model_om", "derived_graph", "qualification"):
        relative = Path(profile[name])
        target = (path.parent / relative).resolve(strict=True)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not target.is_relative_to(path.parent)
        ):
            raise ValueError("speaker artifact escapes its profile bundle")
        files[name] = target
    receipt = json.loads(files["qualification"].read_text("utf-8"))
    if (
        receipt.get("passed") is not True
        or receipt.get("neural_execution") is not True
        or receipt.get("execution_device") != "NPU"
        or receipt.get("input_shape") != [1, 1682, 80]
        or receipt.get("output_shape") != [1, 192]
        or receipt.get("input_features_sha256") != INPUT_SHA256
        or receipt.get("output_sha256") != OUTPUT_SHA256
        or digest(files["model_om"]) != receipt.get("model_om_sha256")
        or digest(files["derived_graph"]) != DERIVED_GRAPH_SHA256
    ):
        raise ValueError("fixed-reference NPU qualification or artifact hash differs")
    return files, receipt


def check(code, operation):
    if code != 0:
        raise RuntimeError(f"ACL {operation} failed: {code}")


class NpuSpeakerSession:
    def __init__(self, original_path):
        profile = os.environ.get("FULL_DUPLEX_NPU_SPEAKER_PROFILE")
        if not profile:
            raise ValueError(
                "FULL_DUPLEX_NPU_SPEAKER_PROFILE must name a qualified bundle"
            )
        files, self.qualification = load_profile(profile)
        self._initialize(original_path, files, self.qualification)

    @classmethod
    def for_reference_qualification(cls, original_path, model_om, derived_graph):
        """Bootstrap only a fixed-reference control, without a prior receipt.

        run() still enforces the exact original input and qualified output hash.
        The serving constructor continues to require a passed NPU receipt.
        """
        files = {
            "model_om": Path(model_om).resolve(strict=True),
            "derived_graph": Path(derived_graph).resolve(strict=True),
        }
        if digest(files["derived_graph"]) != DERIVED_GRAPH_SHA256:
            raise ValueError("qualified fixed-reference derived graph required")
        instance = cls.__new__(cls)
        instance._initialize(
            original_path, files, {"passed": False, "qualification_in_progress": True}
        )
        return instance

    def _initialize(self, original_path, files, qualification):
        self.qualification = qualification
        if digest(Path(original_path)) != ORIGINAL_ONNX_SHA256:
            raise ValueError("original published speaker ONNX differs")
        import acl
        import numpy as np
        import torch
        import torch_npu  # noqa: F401 - registers the NPU backend

        self.acl, self.np, self.torch = acl, np, torch
        self.model_id, self.desc = None, None
        self.device = torch.npu.current_device()
        try:
            model_id, code = acl.mdl.load_from_file(str(files["model_om"]))
            check(code, "load fixed speaker model")
            self.model_id = model_id
            self.desc = acl.mdl.create_desc()
            if not self.desc:
                raise RuntimeError("ACL failed to create speaker model descriptor")
            check(acl.mdl.get_desc(self.desc, model_id), "describe speaker model")
            if (
                acl.mdl.get_num_inputs(self.desc) != 1
                or acl.mdl.get_num_outputs(self.desc) != 1
            ):
                raise ValueError("one fixed input and output required")
            if (
                acl.mdl.get_input_data_type(self.desc, 0) != 0
                or acl.mdl.get_output_data_type(self.desc, 0) != 0
            ):
                raise ValueError("original float32 speaker input and output required")
            self.input_size = acl.mdl.get_input_size_by_index(self.desc, 0)
            self.output_size = acl.mdl.get_output_size_by_index(self.desc, 0)
            dims, code = acl.mdl.get_output_dims(self.desc, 0)
            check(code, "read output dimensions")
            self.output_shape = list(dims["dims"])
            if self.output_shape != [1, 192]:
                raise ValueError("qualified fixed speaker output shape differs")
        except Exception as error:
            try:
                self.close()
            except Exception as cleanup_error:
                error.add_note("speaker initialization cleanup: " + str(cleanup_error))
            raise
        atexit.register(self.close)

    def get_inputs(self):
        return [SimpleNamespace(name="input")]

    def run(self, output_names, inputs):
        acl, np = self.acl, self.np
        if output_names is not None or list(inputs) != ["input"]:
            raise ValueError("original fixed speaker call signature required")
        if self.torch.npu.current_device() != self.device:
            raise ValueError("NPU device changed after speaker initialization")
        data = np.ascontiguousarray(inputs["input"])
        if (
            data.dtype != np.float32
            or list(data.shape) != [1, 1682, 80]
            or data.nbytes != self.input_size
            or hashlib.sha256(data.tobytes()).hexdigest() != INPUT_SHA256
        ):
            raise ValueError("only the original fixed reference features are qualified")
        result = np.empty(self.output_shape, dtype=np.float32)
        if result.nbytes != self.output_size:
            raise ValueError("qualified speaker output byte size differs")
        allocations, datasets = [], []
        try:
            for array, upload in ((data, True), (result, False)):
                pointer, code = acl.rt.malloc(array.nbytes, 0)
                check(code, "allocate speaker tensor")
                allocations.append(pointer)
                if upload:
                    check(
                        acl.rt.memcpy(
                            pointer,
                            array.nbytes,
                            acl.util.numpy_to_ptr(array),
                            array.nbytes,
                            1,
                        ),
                        "copy speaker input to NPU",
                    )
                dataset = acl.mdl.create_dataset()
                if not dataset:
                    raise RuntimeError("ACL failed to create speaker dataset")
                datasets.append((dataset, None))
                buffer = acl.create_data_buffer(pointer, array.nbytes)
                if not buffer:
                    raise RuntimeError("ACL failed to create speaker data buffer")
                datasets[-1] = (dataset, buffer)
                _, code = acl.mdl.add_dataset_buffer(dataset, buffer)
                check(code, "bind speaker tensor")
            check(
                acl.mdl.execute(self.model_id, datasets[0][0], datasets[1][0]),
                "execute speaker on NPU",
            )
            check(
                acl.rt.memcpy(
                    acl.util.numpy_to_ptr(result),
                    result.nbytes,
                    allocations[1],
                    result.nbytes,
                    2,
                ),
                "copy speaker output from NPU",
            )
            if (
                not np.isfinite(result).all()
                or hashlib.sha256(result.tobytes()).hexdigest() != OUTPUT_SHA256
            ):
                raise ValueError(
                    "speaker output differs from the qualified fixed NPU profile"
                )
            return [result]
        finally:
            operations = []
            for dataset, buffer in datasets:
                if buffer is not None:
                    operations.append((acl.destroy_data_buffer, buffer))
                operations.append((acl.mdl.destroy_dataset, dataset))
            for pointer in allocations:
                operations.append((acl.rt.free, pointer))
            errors = []
            for function, handle in operations:
                try:
                    check(function(handle), "release speaker temporary resource")
                except Exception as error:
                    errors.append(str(error))
            if errors:
                active = sys.exception()
                if active is not None:
                    active.add_note("speaker resource cleanup: " + "; ".join(errors))
                else:
                    raise RuntimeError("speaker resource cleanup: " + "; ".join(errors))

    def close(self):
        if self.model_id is not None:
            self.torch.npu.set_device(self.device)
            check(self.acl.mdl.unload(self.model_id), "unload speaker model")
            self.model_id = None
        if self.desc is not None:
            check(self.acl.mdl.destroy_desc(self.desc), "destroy speaker descriptor")
            self.desc = None
