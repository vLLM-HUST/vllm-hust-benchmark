# Embedded Pipeline evidence

These are unchanged copies of two measured records previously embedded under dev-hub's
`scripts/frontier_pipeline/` directory. `manifest.json` records their source commit, original paths,
SHA256 and lengths.

`results.json` is the original paired measurement summary. `ecpa-receipt.json` is the original
real-online smoke receipt, not a formal performance claim. No points are combined, recomputed or
promoted by archiving them here.

Source cleanup was merged in [dev-hub #99](https://github.com/vLLM-HUST/vllm-hust-dev-hub/pull/99)
after custody and Quickstart checks passed. The two duplicate JSON files were removed; receipt tests
now fetch the immutable benchmark copy and verify its SHA256 and byte length before retaining the
measured-policy assertions. Other evidence and consumer migrations remain tracked in benchmark issue
#326.
