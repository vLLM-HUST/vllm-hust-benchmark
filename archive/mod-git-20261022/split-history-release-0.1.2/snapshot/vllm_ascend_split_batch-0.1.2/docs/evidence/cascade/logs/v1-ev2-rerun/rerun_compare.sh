#!/bin/bash
# Evidence #2 re-run comparator driver.  Reuses ev2_compare.py and
# ev2_natural_prefix.py VERBATIM (no harness edit); only the file names differ.
OUT=/vllm-workspace/flashinfer-migration/cascade-evidence/logs/v1-ev2-rerun
D="$OUT/raw_dumps"
CMP=/vllm-workspace/flashinfer-migration/cascade-evidence/ev2_compare.py
NAT=/vllm-workspace/flashinfer-migration/cascade-evidence/ev2_natural_prefix.py
cd /tmp
{
echo "############ Evidence #2 RE-RUN comparisons (fixed build ddc0120, unshimmed) ############"
echo "############ comparator: ev2_compare.py (byte-identical口径 to c3_compare.py) ############"
echo
echo "### LANE G (graph, UNSHIMMED) B=64 shared~4356 gen128 ignore_eos ###"
python "$CMP" "$D/ev2_on_gB64_ie_on.json"     "$D/ev2_off_gB64_ie_off.json"     "G-ie B64 graph on(bf16) vs off  [MAIN ACCEPTANCE]"
python "$CMP" "$D/ev2_on_run2_gB64_ie_on_run2.json" "$D/ev2_on_gB64_ie_on.json"  "G-ie B64 graph on_run2 vs on  [DETERMINISM]"
python "$CMP" "$D/ev2_off_gB64_ie_off.json"   "$D/ev2_off_gB64_ie_off2.json"    "G-ie B64 graph off vs off2    [NOISE FLOOR]"
python "$CMP" "$D/ev2_on_fp32_gB64_ie_on_fp32.json" "$D/ev2_on_gB64_ie_on.json" "G-ie B64 graph on_fp32 vs on(bf16)"
python "$CMP" "$D/ev2_on_fp32_gB64_ie_on_fp32.json" "$D/ev2_off_gB64_ie_off.json" "G-ie B64 graph on_fp32 vs off"
echo
echo "### LANE G B=32 ###"
python "$CMP" "$D/ev2_on_gB32_ie_on.json"     "$D/ev2_off_gB32_ie_off.json"     "G-ie B32 graph on(bf16) vs off"
echo
echo "### LANE G literal C3 config (graph, UNSHIMMED, no ignore_eos) ###"
python "$CMP" "$D/ev2_on_gB64_lit_on.json"    "$D/ev2_off_gB64_lit_off.json"    "G-lit B64 graph on(bf16) vs off"
python "$CMP" "$D/ev2_on_run2_gB64_lit_on_run2.json" "$D/ev2_on_gB64_lit_on.json" "G-lit B64 graph on_run2 vs on [DETERMINISM]"
python "$CMP" "$D/ev2_on_gB32_lit_on.json"    "$D/ev2_off_gB32_lit_off.json"    "G-lit B32 graph on(bf16) vs off"
echo
echo "### LANE G extra: trace-invariance of the determinism leg ###"
python "$CMP" "$D/ev2_on_run2_gB64_ie_on_run2_trace.json" "$D/ev2_on_gB64_ie_on.json" "G-ie B64 graph on_run2(trace) vs on(trace) [DETERMINISM, both traced]"
python "$CMP" "$D/ev2_on_run2_gB64_ie_on_run2_trace.json" "$D/ev2_on_run2_gB64_ie_on_run2.json" "G-ie B64 graph on_run2(trace) vs on_run2(no-trace) [TRACE INVARIANCE]"
echo
echo "### LANE E (eager) B=64 shared~4356 gen128 ignore_eos ###"
python "$CMP" "$D/ev2_on_eB64_ie_on.json"     "$D/ev2_off_eB64_ie_off.json"     "E-ie B64 eager on(bf16) vs off"
python "$CMP" "$D/ev2_on_run2_eB64_ie_on_run2.json" "$D/ev2_on_eB64_ie_on.json"  "E-ie B64 eager on_run2 vs on [DETERMINISM]"
python "$CMP" "$D/ev2_off_eB64_ie_off.json"   "$D/ev2_off_eB64_ie_off2.json"    "E-ie B64 eager off vs off2   [NOISE FLOOR]"
python "$CMP" "$D/ev2_on_fp32_eB64_ie_on_fp32.json" "$D/ev2_off_eB64_ie_off.json" "E-ie B64 eager on_fp32 vs off"
python "$CMP" "$D/ev2_on_fp32_eB64_ie_on_fp32.json" "$D/ev2_on_eB64_ie_on.json" "E-ie B64 eager on_fp32 vs on(bf16)"
python "$CMP" "$D/ev2_on_eB64_ie_on_torchmerge.json" "$D/ev2_on_eB64_ie_on.json" "E-ie B64 eager torchmerge-on vs kernelmerge-on [ABLATION]"
python "$CMP" "$D/ev2_on_eB64_ie_on_torchmerge.json" "$D/ev2_off_eB64_ie_off.json" "E-ie B64 eager torchmerge-on vs off"
echo
echo "### LANE E literal C3 config (no ignore_eos, stand-in EOS-diluted) ###"
python "$CMP" "$D/ev2_on_eB64_lit_on.json"    "$D/ev2_off_eB64_lit_off.json"    "E-lit B64 eager on(bf16) vs off"
echo
echo "### LANE E B=32 ###"
python "$CMP" "$D/ev2_on_eB32_ie_on.json"     "$D/ev2_off_eB32_ie_off.json"     "E-ie B32 eager on(bf16) vs off"
echo
echo "### CROSS-MODE controls (diagnostic only) ###"
python "$CMP" "$D/ev2_off_eB64_ie_off.json"   "$D/ev2_off_gB64_ie_off.json"     "CTRL B64 eager-off vs graph-off"
python "$CMP" "$D/ev2_on_eB64_ie_on.json"     "$D/ev2_on_gB64_ie_on.json"       "CTRL B64 eager-on vs graph-on"
} > "$OUT/comparisons_raw.txt" 2>&1
echo "wrote $OUT/comparisons_raw.txt"

{
echo "############ Natural (pre-EOS) vs post-EOS divergence locus ############"
echo "### LANE G B=64 [MAIN] ###"
python "$NAT" "$D/ev2_off_gB64_ie_off.json" "$D/ev2_on_gB64_ie_on.json" "G-ie B64 on vs off"
echo "### LANE G B=32 ###"
python "$NAT" "$D/ev2_off_gB32_ie_off.json" "$D/ev2_on_gB32_ie_on.json" "G-ie B32 on vs off"
echo "### LANE G literal B=64 ###"
python "$NAT" "$D/ev2_off_gB64_lit_off.json" "$D/ev2_on_gB64_lit_on.json" "G-lit B64 on vs off"
echo "### LANE G literal B=32 ###"
python "$NAT" "$D/ev2_off_gB32_lit_off.json" "$D/ev2_on_gB32_lit_on.json" "G-lit B32 on vs off"
echo "### LANE E B=64 ###"
python "$NAT" "$D/ev2_off_eB64_ie_off.json" "$D/ev2_on_eB64_ie_on.json" "E-ie B64 on vs off"
echo "### LANE E B=32 ###"
python "$NAT" "$D/ev2_off_eB32_ie_off.json" "$D/ev2_on_eB32_ie_on.json" "E-ie B32 on vs off"
echo "### LANE E literal ###"
python "$NAT" "$D/ev2_off_eB64_lit_off.json" "$D/ev2_on_eB64_lit_on.json" "E-lit B64 on vs off"
} > "$OUT/natural_prefix_analysis.txt" 2>&1
echo "wrote $OUT/natural_prefix_analysis.txt"
head -40 "$OUT/comparisons_raw.txt"
