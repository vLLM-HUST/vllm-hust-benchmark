#!/bin/bash
# Re-run every comparison and archive the raw comparator output.
OUT=/vllm-workspace/flashinfer-migration/cascade-evidence/logs/v1-ev2
CMP=/vllm-workspace/flashinfer-migration/cascade-evidence/ev2_compare.py
cd /tmp
{
echo "############ Evidence #2 comparisons (all raw) ############"
echo "### Lane A literal-C3-config (no ignore_eos; diluted) ###"
python "$CMP" ev2_off_eager_off.json ev2_on_eager_on.json "A-lit B64 eager on(bf16) vs off"
python "$CMP" ev2_on_eager_on.json ev2_on_run2_eager_on_run2.json "A-lit B64 eager on_run2 vs on"
python "$CMP" ev2_on_eager_on.json ev2_on_fp32_eager_on_fp32.json "A-lit B64 eager on_fp32 vs on(bf16)"
python "$CMP" ev2_off_eager_off.json ev2_on_fp32_eager_on_fp32.json "A-lit B64 eager on_fp32 vs off"
python "$CMP" ev2_on_eager_on.json ev2_on_eager_on_torchmerge.json "A-lit B64 eager torchmerge-on vs kernelmerge-on"
echo
echo "### Lane A ignore_eos full 128-step decode ###"
python "$CMP" ev2_off_eagerB64_ie_off.json ev2_on_eagerB64_ie_on.json "A-ie B64 eager on(bf16) vs off"
python "$CMP" ev2_off_eagerB64_ie_off.json ev2_on_fp32_eagerB64_ie_on_fp32.json "A-ie B64 eager on_fp32 vs off"
python "$CMP" ev2_on_eagerB64_ie_on.json ev2_on_fp32_eagerB64_ie_on_fp32.json "A-ie B64 eager on_fp32 vs on(bf16)"
python "$CMP" ev2_on_eagerB64_ie_on.json ev2_on_run2_eagerB64_ie_on_run2.json "A-ie B64 eager on_run2 vs on (determinism)"
python "$CMP" ev2_off_eagerB64_ie_off.json ev2_off_eagerB64_ie_off2.json "A-ie B64 eager off vs off2 (determinism)"
python "$CMP" ev2_on_eagerB64_ie_on.json ev2_on_eagerB64_ie_on_torchmerge.json "A-ie B64 eager torchmerge-on vs kernelmerge-on (ablation)"
python "$CMP" ev2_off_eagerB64_ie_off.json ev2_on_eagerB64_ie_on_torchmerge.json "A-ie B64 eager torchmerge-on vs off"
echo
echo "### Lane A B=32 ###"
python "$CMP" ev2_off_eagerB32_ie_off.json ev2_on_eagerB32_ie_on.json "A-ie B32 eager on(bf16) vs off"
echo
echo "### Reference-frame control ###"
python "$CMP" ev2_off_eagerB64_ie_off.json ev2_off_graphB64_ie_off.json "CTRL B64 eager-off vs graph-off"
echo
echo "### Lane B/C graph twin (shim = DIAGNOSTIC ONLY) ###"
python "$CMP" ev2_off_shimB64_ie_off.json ev2_on_shimB64_ie_on.json "C-shim B64 graph on vs off"
python "$CMP" ev2_off_shimB64_ie_off.json ev2_on_traceB64_ie_on.json "C-shim B64 graph on(trace) vs off"
python "$CMP" ev2_off_shimB64_ie_off.json ev2_off_shimB64_ie_off2.json "C-shim B64 graph off vs off2 (determinism)"
python "$CMP" ev2_off_eagerB64_ie_off.json ev2_off_shimB64_ie_off.json "CTRL B64 eager-off vs shim-graph-off"
} > "$OUT/comparisons_raw.txt" 2>&1
echo "wrote $OUT/comparisons_raw.txt"
cat "$OUT/comparisons_raw.txt" | head -20
