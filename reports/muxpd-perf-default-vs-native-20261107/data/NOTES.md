
- 2026-11-07: first launch of mux43-tp2 aborted before any plan ran (keepalive restarted during server load by a
  faulty watcher); kept as runs/mux43-tp2-aborted-keepalive. Relaunched unchanged. Plan text above not otherwise edited.
- 2026-11-07 07:06 UTC: plan revision 1 (PLAN-REVISION-1.md). Old driver run.sh stopped during TP2 MuxPD pass1 after
  burst-s1, burst-s2; in-flight contend-s1 left to finish; passes 2-3 never ran. Keepalive watcher moved into run2.sh.
