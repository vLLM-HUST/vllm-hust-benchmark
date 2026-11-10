"""Metadata-only RPC outside timed generation; no per-step instrumentation."""
def install():
    import dataclasses
    from vllm_ascend.worker.worker import NPUWorker
    def witness(worker):
        r=worker.model_runner;d=r.cudagraph_dispatcher
        return {'rank':worker.rank,'proposer':type(r.drafter).__name__,
                'uniform_decode_query_len':r.uniform_decode_query_len,
                'resolved_capture_sizes':r.compilation_config.cudagraph_capture_sizes,
                'graph_keys':{str(k):[dataclasses.asdict(v) for v in sorted(values,key=repr)] for k,values in d.cudagraph_keys.items()}}
    NPUWorker.gamma4_runtime_witness=witness
