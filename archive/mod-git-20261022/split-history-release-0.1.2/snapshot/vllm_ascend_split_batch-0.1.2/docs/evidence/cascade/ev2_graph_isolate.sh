#!/bin/bash
# Isolate the graph-ON engine-init failure.
#
# NOTE: the harness only sets VLLM_ASCEND_ENABLE_CASCADE_GRAPH in "on*" modes, so
# the earlier G1-G3 diagnostics (mode=off) actually ran with graph_gate=0 and are
# NOT valid isolations.  Here the graph gate is exported directly and the legs
# differ ONLY in which plugin install step is neutralized.
#
# All legs use a fresh VLLM_CACHE_ROOT (the shared /root/.cache/vllm AOT entry
# proved unstable: an identical-hash entry both loaded and failed across runs).
set -x
OUT=/vllm-workspace/flashinfer-migration/cascade-evidence/logs/v1-ev2
RUN=/vllm-workspace/flashinfer-migration/cascade-evidence/ev2_run.py
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate hust
cd /tmp
export ASCEND_RT_VISIBLE_DEVICES=6 HF_HUB_OFFLINE=1

leg() {  # tag extra_env...
  local tag=$1; shift
  rm -rf "/tmp/ev2_cache_$tag"
  env VLLM_CACHE_ROOT="/tmp/ev2_cache_$tag" \
      VLLM_ASCEND_ENABLE_CASCADE_GRAPH=1 \
      EV2_BATCH=4 EV2_SENT=420 EV2_GEN=8 EV2_MAXLEN=8192 \
      EV2_EAGER=0 EV2_CAPTURE=32,64 "$@" \
      python -u "$RUN" off "$tag" > "$OUT/$tag.log" 2>&1
  echo "rc=$? tag=$tag"
  grep -m1 "cascade plugin loaded" "$OUT/$tag.log"
  grep -m1 "NoneType' object is not callable" "$OUT/$tag.log"
  grep -m1 "Graph capturing finished\|Engine core initialization failed" "$OUT/$tag.log"
}

echo "##### G3': graph_gate=1, BOTH plugin installs neutralized (baseline control) #####"
leg g3b_gate1_skip_both EV2_SKIP_GRAPH_INSTALL=1 EV2_SKIP_RUNNER_PATCH=1

echo "##### G1': graph_gate=1, cascade_graph_plugin.install neutralized #####"
leg g1b_gate1_skip_graph_install EV2_SKIP_GRAPH_INSTALL=1

echo "##### G2': graph_gate=1, cascade_runner_patch.install neutralized #####"
leg g2b_gate1_skip_runner_patch EV2_SKIP_RUNNER_PATCH=1

echo "##### G4: graph_gate=1, both installs active (full plugin) #####"
leg g4_gate1_full

echo "##### DONE #####"
