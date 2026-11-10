"""Private API spelling adapter for the recorded ascend87 stack; no target changes."""


def patch_mtp_group_hook(module, annotate):
    """Attach unchanged Qwen MTP annotation to the replacement group-builder API."""
    import inspect
    from functools import wraps

    original = module.get_kv_cache_groups
    assert list(inspect.signature(original).parameters) == [
        "vllm_config",
        "kv_cache_spec",
    ]
    if getattr(original, "_bench_q35_group_hook", False):
        return False

    @wraps(original)
    def get_groups(vllm_config, kv_cache_spec):
        groups = original(vllm_config, kv_cache_spec)
        annotate(vllm_config, kv_cache_spec, groups)
        return groups

    get_groups._bench_q35_group_hook = True
    module.get_kv_cache_groups = get_groups
    return True


def install():
    import inspect
    import os

    if os.environ.get("HUST_VSPEC_METHOD") == "mtp":
        from vllm.v1.core import kv_cache_utils

        if not hasattr(kv_cache_utils, "_annotate_eagle_groups"):
            from vllm_hust_vspec.backends import mtp

            # The new host annotates only DeepSeek inline; no generic hook remains.
            # Keep its group builder and the plugin's exact Qwen annotation logic.
            mtp._patch_mtp_hybrid_kv_group_annotation = lambda: patch_mtp_group_hook(
                kv_cache_utils, mtp._annotate_qwen35_mtp_kv_groups
            )
        from vllm.v1.worker import gpu_model_runner, mamba_utils

        if not hasattr(gpu_model_runner.GPUModelRunner, "_get_mamba_state_copy_funcs"):
            # Host now owns per-plane copying in MambaBuffers/collect_mamba_copy_meta.
            # Installing the old dict-returning group API would corrupt the new tuple contract.
            assert not hasattr(gpu_model_runner, "copy_kv_cache_blocks_inplace")
            assert hasattr(gpu_model_runner.GPUModelRunner, "_get_mamba_bufs")
            assert hasattr(mamba_utils.MambaBuffers, "create")
            assert (
                list(
                    inspect.signature(
                        mamba_utils.postprocess_mamba_align_gpu
                    ).parameters
                )[0]
                == "bufs"
            )
            assert str(
                inspect.signature(mamba_utils.get_mamba_groups).return_annotation
            ).startswith("tuple[")
            from vllm_hust_vspec.backends import mtp

            mtp._patch_mtp_mamba_group_api = lambda: False
            print(
                "BENCH: retaining native MambaBuffers tuple/group and per-plane copy API; obsolete legacy compatibility shim not installed",
                flush=True,
            )
    from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer

    cls = AscendSpecDecodeBaseProposer
    if not hasattr(cls, "_sample_draft_from_logits"):
        native = cls._sample_from_logits
        assert list(inspect.signature(native).parameters) == [
            "self",
            "logits",
            "sampling_metadata",
        ]
        cls._sample_draft_from_logits = native
    # The exact Qwen2.5 pair has byte-for-byte equal tokenizer JSON but different
    # padded LM-head row counts. Permit only this validated shared-ID contract.
    import json
    from pathlib import Path
    from vllm.config.speculative import SpeculativeConfig

    if getattr(SpeculativeConfig, "_bench_shared_padding", False):
        return
    original = SpeculativeConfig.verify_equal_vocab_size_if_draft_model

    def verify(self):
        try:
            return original(self)
        except ValueError:
            target = self.target_model_config
            draft = self.draft_model_config
            if self.method != "draft_model" or target is None or draft is None:
                raise
            if (Path(target.model).name, Path(draft.model).name) != (
                "Qwen2.5-14B-Instruct",
                "Qwen2.5-0.5B-Instruct",
            ):
                raise
            a = json.loads((Path(target.model) / "tokenizer.json").read_text())
            b = json.loads((Path(draft.model) / "tokenizer.json").read_text())
            if a != b:
                raise
            ids = [*a["model"]["vocab"].values(), *(x["id"] for x in a["added_tokens"])]
            if min(ids) < 0 or max(ids) >= min(
                target.get_vocab_size(), draft.get_vocab_size()
            ):
                raise
            # Do not resize heads, remap tokens, or mask target logits. A generated
            # padding ID outside draft embeddings remains a hard runtime failure.

    SpeculativeConfig.verify_equal_vocab_size_if_draft_model = verify
    SpeculativeConfig._bench_shared_padding = True
    # Adaptive changes the initial gamma after the runner was constructed. The
    # host initializes its cached reorder threshold from that earlier gamma.
    # Recompute the cached value from ALL current builders; never waive an
    # actual disagreement between backends.
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    if not getattr(NPUModelRunner, "_bench_recompute_reorder", False):
        original_reorder = NPUModelRunner.calculate_reorder_batch_threshold

        def calculate_reorder(self):
            thresholds = {
                getattr(g.get_metadata_builder(), "reorder_batch_threshold", None)
                for g in self._attn_group_iterator()
            } - {None}
            if len(thresholds) == 1 and self.reorder_batch_threshold not in thresholds:
                print(
                    f"BENCH reorder cache refresh: old={self.reorder_batch_threshold} builders={thresholds}",
                    flush=True,
                )
                self.reorder_batch_threshold = None
            return original_reorder(self)

        NPUModelRunner.calculate_reorder_batch_threshold = calculate_reorder
        NPUModelRunner._bench_recompute_reorder = True
    if not getattr(NPUModelRunner, "_bench_dynamic_draft_rows", False):
        original_rows = NPUModelRunner._get_draft_token_ids_cpu

        def get_rows(self):
            rows, req_ids = original_rows(self)
            draft = self._draft_token_ids
            if hasattr(draft, "shape") and len(draft.shape) == 2:
                rows = trim_dynamic_rows(rows, int(draft.shape[1]))
            return rows, req_ids

        NPUModelRunner._get_draft_token_ids_cpu = get_rows
        NPUModelRunner._bench_dynamic_draft_rows = True


def trim_dynamic_rows(rows, width):
    """Restore actual proposal width from padded D2H storage, never drop real IDs."""
    for row in rows:
        if len(row) < width or any(token != -1 for token in row[width:]):
            raise RuntimeError("Dynamic draft width conflicts with real token IDs")
    return [row[:width] for row in rows]
