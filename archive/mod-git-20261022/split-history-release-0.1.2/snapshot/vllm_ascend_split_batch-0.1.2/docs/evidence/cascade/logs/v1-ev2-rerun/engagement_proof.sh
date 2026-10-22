#!/bin/bash
# Item 5: per-leg activation + fail-open accounting.  Pure log analysis.
OUT=/vllm-workspace/flashinfer-migration/cascade-evidence/logs/v1-ev2-rerun
cd "$OUT"
for f in gB64_ie_off gB64_ie_on gB64_ie_on_run2 gB64_ie_off2 gB64_ie_on_fp32 \
         gB32_ie_off gB32_ie_on \
         gB64_lit_off gB64_lit_on gB64_lit_on_run2 gB32_lit_off gB32_lit_on \
         gB64_ie_on_run2_trace \
         eB64_ie_off eB64_ie_on eB64_ie_on_run2 eB64_ie_off2 eB64_ie_on_fp32 \
         eB64_ie_on_torchmerge eB64_lit_off eB64_lit_on eB32_ie_off eB32_ie_on; do
  [ -f "$f.log" ] || continue
  echo "===== $f ====="
  echo -n "  gate_marker: "; grep -m1 -o 'cascade plugin loaded (gate=[01], graph_gate=[01], kernel_wheel=[a-z]*)' "$f.log" || echo "(none)"
  echo "  cascade-active           : $(grep -c 'cascade-active' "$f.log")"
  echo "  capture body SUCCESS     : $(grep -c 'capture body SUCCESS' "$f.log")"
  grep -o 'capture body SUCCESS: num_tokens=[0-9]*' "$f.log" | sort | uniq -c | sed 's/^/      /'
  echo "  replay update key hit    : $(grep -c 'replay update: cascade key hit' "$f.log")"
  grep -o 'replay update: cascade key hit num_tokens=[0-9]*' "$f.log" | sort | uniq -c | sed 's/^/      /'
  echo "  replay_swap True/False   : $(grep -c 'replay_swap=True' "$f.log") / $(grep -c 'replay_swap=False' "$f.log")"
  echo "  stage1 re-bind skipped   : $(grep -c 'stage1 re-bind skipped' "$f.log")"
  echo "  twin-miss WARNING        : $(grep -c 'cascade twin missing for descriptor' "$f.log")"
  grep -o 'twin missing for descriptor BatchDescriptor([^)]*)' "$f.log" | sort | uniq -c | sed 's/^/      /'
  echo "  twin-miss trace lines    : $(grep -c 'twin miss desc=' "$f.log")"
  echo "  drift fail-open WARNING  : $(grep -c 'delegating to the original implementation' "$f.log")"
  echo "  capture fell back        : $(grep -c 'cascade graph capture fell back' "$f.log")"
  echo "  GraphParams unavailable  : $(grep -c 'GraphParams unavailable' "$f.log")"
  echo "  param lists misaligned   : $(grep -c 'param lists misaligned' "$f.log")"
  echo "  replay w/o shared_len    : $(grep -c 'replay without cascade_shared_len' "$f.log")"
  echo "  replay short real req    : $(grep -c 'replay with short real request' "$f.log")"
  echo "  replay update failed     : $(grep -c 'cascade replay update failed' "$f.log")"
  echo "  wheel unavailable        : $(grep -c 'ascend_kernel wheel unavailable' "$f.log")"
  echo "  lse_merge failed         : $(grep -c 'lse_merge kernel failed' "$f.log")"
  echo "  bf16-difference notice   : $(grep -c 'bf16-level difference versus the standard full-KV FIA path' "$f.log")"
  echo "  TypeError / Traceback    : $(grep -c 'TypeError' "$f.log") / $(grep -c 'Traceback' "$f.log")"
  echo -n "  EV2 summary: "; grep -h '^EV2 mode=' "$f.log" || echo "(no EV2 line)"
done
