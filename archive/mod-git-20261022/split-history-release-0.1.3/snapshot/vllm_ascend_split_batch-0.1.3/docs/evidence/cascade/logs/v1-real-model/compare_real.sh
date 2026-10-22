#!/bin/bash
# Evidence #2 real-model comparator driver.  Reuses ev2_compare.py (parent dir,
# VERBATIM) and ev2_natural_prefix_real.py (verbatim except the tokenizer path;
# EOS ids identical to the stand-in).  Only file names differ.
OUT=/vllm-workspace/knowledge/evidence/cascade/logs/v1-real-model
D="$OUT/raw_dumps"
CMP=/vllm-workspace/knowledge/evidence/cascade/ev2_compare.py
NAT=$OUT/ev2_natural_prefix_real.py
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate hust
cd /tmp
export HF_HUB_OFFLINE=1

{
echo "############ Evidence #2 REAL-MODEL re-run (Qwen2.5-14B-Instruct, natural EOS) ############"
echo "############ comparator: ev2_compare.py (byte-identical口径 to c3_compare.py) ############"
echo
echo "### LANE G (graph) B=64 shared~4224 gen128 natural EOS ###"
python "$CMP" "$D/ev2_on_gB64_on.json"     "$D/ev2_off_gB64_off.json"     "G B64 graph on(bf16) vs off  [MAIN ACCEPTANCE]"
python "$CMP" "$D/ev2_on_run2_gB64_on_run2.json" "$D/ev2_on_gB64_on.json"  "G B64 graph on_run2 vs on    [DETERMINISM]"
echo
echo "### LANE G B=32 ###"
python "$CMP" "$D/ev2_on_gB32_on.json"     "$D/ev2_off_gB32_off.json"     "G B32 graph on(bf16) vs off"
echo
echo "### LANE E (eager, bf16 reference frame) B=64 ###"
python "$CMP" "$D/ev2_on_eB64_on.json"     "$D/ev2_off_eB64_off.json"     "E B64 eager on(bf16) vs off"
python "$CMP" "$D/ev2_off_eB64_off.json"   "$D/ev2_off_eB64_off2.json"    "E B64 eager off vs off2      [NOISE FLOOR]"
echo
echo "### CROSS-MODE control (diagnostic only) ###"
python "$CMP" "$D/ev2_off_eB64_off.json"   "$D/ev2_off_gB64_off.json"     "CTRL B64 eager-off vs graph-off"
} > "$OUT/comparisons_raw.txt" 2>&1
echo "wrote $OUT/comparisons_raw.txt"

{
echo "############ Natural (pre-EOS) vs post-EOS divergence locus ############"
echo "### LANE G B=64 [MAIN] ###"
python "$NAT" "$D/ev2_off_gB64_off.json" "$D/ev2_on_gB64_on.json" "G B64 on vs off"
echo "### LANE G B=32 ###"
python "$NAT" "$D/ev2_off_gB32_off.json" "$D/ev2_on_gB32_on.json" "G B32 on vs off"
echo "### LANE E B=64 ###"
python "$NAT" "$D/ev2_off_eB64_off.json" "$D/ev2_on_eB64_on.json" "E B64 on vs off"
echo "### NOISE FLOOR off vs off2 (eager) ###"
python "$NAT" "$D/ev2_off_eB64_off.json" "$D/ev2_off_eB64_off2.json" "E B64 off vs off2"
} > "$OUT/natural_prefix_analysis.txt" 2>&1
echo "wrote $OUT/natural_prefix_analysis.txt"
cat "$OUT/comparisons_raw.txt"
