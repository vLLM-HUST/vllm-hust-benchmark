#!/bin/bash
# Isolate the graph-lane engine-init failure (NoneType not callable).
# Runs after ev2_eager_matrix.sh has finished (card 6 is single-tenant).
set -x
OUT=/vllm-workspace/flashinfer-migration/cascade-evidence/logs/v1-ev2
RUN=/vllm-workspace/flashinfer-migration/cascade-evidence/ev2_run.py
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate hust
cd /tmp
export ASCEND_RT_VISIBLE_DEVICES=6 HF_HUB_OFFLINE=1

echo "##### G1: graph gate ON, graph_plugin.install neutralized #####"
EV2_BATCH=4 EV2_SENT=420 EV2_GEN=8 EV2_MAXLEN=8192 EV2_EAGER=0 \
EV2_GRAPH=1 EV2_CAPTURE=32,64 EV2_SKIP_GRAPH_INSTALL=1 \
  python -u "$RUN" off g1_skip_graph_install > "$OUT/diag_g1_skip_graph_install.log" 2>&1
echo "rc=$? g1"; tail -2 "$OUT/diag_g1_skip_graph_install.log"

echo "##### G2: graph gate ON, runner_patch.install neutralized #####"
EV2_BATCH=4 EV2_SENT=420 EV2_GEN=8 EV2_MAXLEN=8192 EV2_EAGER=0 \
EV2_GRAPH=1 EV2_CAPTURE=32,64 EV2_SKIP_RUNNER_PATCH=1 \
  python -u "$RUN" off g2_skip_runner_patch > "$OUT/diag_g2_skip_runner_patch.log" 2>&1
echo "rc=$? g2"; tail -2 "$OUT/diag_g2_skip_runner_patch.log"

echo "##### G3: graph gate ON, both installs neutralized (pure baseline) #####"
EV2_BATCH=4 EV2_SENT=420 EV2_GEN=8 EV2_MAXLEN=8192 EV2_EAGER=0 \
EV2_GRAPH=1 EV2_CAPTURE=32,64 EV2_SKIP_GRAPH_INSTALL=1 EV2_SKIP_RUNNER_PATCH=1 \
  python -u "$RUN" off g3_skip_both > "$OUT/diag_g3_skip_both.log" 2>&1
echo "rc=$? g3"; tail -2 "$OUT/diag_g3_skip_both.log"
echo "##### DIAG DONE #####"
