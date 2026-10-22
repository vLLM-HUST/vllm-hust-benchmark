"""DIAGNOSTIC-ONLY harness shim.

Purpose: answer the question "if the three known host-signature drifts were
fixed, could the graph-twin path run on the v1 baseline at all?"  Results
obtained with it are reported in a separate, clearly labeled section of
section2-correctness.md.  This is NOT an acceptance leg and must not be read as
evidence that the shipped plugin works.

Three drifts, each verified against the host source (see the report):

  D1  plugin `cascade_runner_patch._model_forward` calls
      `self._update_full_graph_params_if_needed(forward_context,
      num_tokens_padded, positions)` -- 3 args.
      v1 host `NPUModelRunner._update_full_graph_params_if_needed` takes 2
      (upstream dropped the `positions` parameter).

  D2  plugin `cascade_runner_patch._patch_capture_scheduling` installs
      `_capture_cudagraphs(self, batch_descriptors, cudagraph_runtime_mode)`.
      v1 core `GPUModelRunner.capture_model` calls it with `profiler=`.

  D3  plugin `cascade_graph_plugin.install` installs an `update_graph_params`
      wrapper whose fallback calls the host implementation with 7 positional
      args (including `num_dcp_pcp_tokens`).
      v1 host `AscendAttentionBackendImpl.update_graph_params` takes 6.

Install BEFORE `from vllm import LLM`: the shim wraps the two plugin install
functions, so the D3 fix is applied before the plugin captures the host
`update_graph_params`, and the D1/D2 fixes are applied right after the plugin
installs its own wrappers (which would otherwise clobber a pre-installed fix).
"""


def install() -> None:
    import vllm_ascend_split_batch.cascade_graph_plugin as gp
    import vllm_ascend_split_batch.cascade_runner_patch as rp

    # ---------------- D3: applied around cascade_graph_plugin.install --------
    orig_gp_install = gp.install

    def gp_install(attn_mod, builder_cls, impl_cls):
        host_update = impl_cls.update_graph_params

        def tolerant_update(
            update_stream,
            forward_context,
            num_tokens,
            vllm_config,
            speculative_config=None,
            num_dcp_pcp_tokens=None,  # dropped by the v1 host
            draft_attn_metadatas=None,
        ):
            return host_update(
                update_stream,
                forward_context,
                num_tokens,
                vllm_config,
                speculative_config,
                draft_attn_metadatas,
            )

        impl_cls.update_graph_params = staticmethod(tolerant_update)
        try:
            return orig_gp_install(attn_mod, builder_cls, impl_cls)
        finally:
            pass

    gp.install = gp_install

    # ---------------- D1 + D2: applied after cascade_runner_patch.install ----
    orig_rp_install = rp.install

    def rp_install():
        ok = orig_rp_install()
        from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

        # D1: tolerate the extra `positions` argument the plugin passes.
        plugin_update = NPUModelRunner._update_full_graph_params_if_needed

        def _update_full_graph_params_if_needed(
            self, forward_context, num_tokens_padded, *args, **kwargs
        ):
            return plugin_update(self, forward_context, num_tokens_padded)

        NPUModelRunner._update_full_graph_params_if_needed = (
            _update_full_graph_params_if_needed
        )

        # D2: tolerate the `profiler` keyword the v1 core passes.
        plugin_capture = NPUModelRunner._capture_cudagraphs

        def _capture_cudagraphs(
            self, batch_descriptors, cudagraph_runtime_mode, profiler=None
        ):
            return plugin_capture(
                self,
                batch_descriptors=batch_descriptors,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
            )

        NPUModelRunner._capture_cudagraphs = _capture_cudagraphs
        return ok

    rp.install = rp_install

    print(
        "[ev2-shim] DIAGNOSTIC-ONLY signature shim installed "
        "(D1 _update_full_graph_params_if_needed, D2 _capture_cudagraphs, "
        "D3 update_graph_params)",
        flush=True,
    )
