# MuxPD scheduling follow-up, 2026-10-26

The finite TP4 performance and lifecycle runs have finished. [REPORT-v2.md](REPORT-v2.md) reports
the asynchronous candidate's throughput recovery, shorter maximum streaming pause, remaining
TTFT/P99 costs, and the bounded lifecycle checks. The quality limitation remains unresolved. The
earlier [preparation draft](REPORT-DRAFT-v1.md) is retained as a historical cutoff.

Publication status was corrected on 2026-10-28 in
[PUBLICATION-STATUS-CORRECTION-v1.md](PUBLICATION-STATUS-CORRECTION-v1.md). The pending-upload
wording retained in REPORT-v2 is stale editorial text: the recorded 2026-10-26 anonymous download
and 212-member checks passed. The original report and evidence bytes are preserved; this correction
changes no performance or quality result.

Nine timed processes and one separate three-case lifecycle process pass. The candidate is within
approximately 1.3% of the fresh native throughput reference, while its worst mixed seed pauses are
about 85% shorter. It does not establish a native-relative throughput win, HTTP Goodput,
Manager/default qualification, or semantic-quality acceptance.

Original run artifacts are published in the
[benchmark Release](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/tag/evidence-muxpd-scheduling-20261026).
They are not stored as payloads in this Git directory or in the plugin repository. See the
[anonymous download receipt](public-download-verification-v1.json),
[member verification](public-member-verification-v1.json), and [member manifest](MANIFEST-v1.json).

The [packager](package_evidence_v1.py) selects only each terminal run's controller/config/workload,
copied producers, results, process log and before/ after NPU inventories, plus this campaign's
benchmark producers, analyses and CPU support logs. It explicitly excludes compiler caches. It
preserves failed terminal runs and does not turn them into successful measurements.

`plan` reads controller JSON and file metadata only. Run `build` only after all timed work finishes;
it requires every observed run to be terminal. The output directory must be fresh and outside a Git
worktree:

```bash
python reports/muxpd-scheduling-20261026/package_evidence_v1.py plan \
  --root /root/workspace/muxpd-scheduling-20261026
python reports/muxpd-scheduling-20261026/package_evidence_v1.py build \
  --root /root/workspace/muxpd-scheduling-20261026 \
  --output /root/workspace/muxpd-scheduling-20261026/publication-v1 \
  --support-dir bench --support-dir bench-v2 --support-dir analysis \
  --support-dir async-prototype --support-dir lifecycle-v1 \
  --support-dir docs-validation --support-dir closure-validation \
  --support-dir publication-tools
```

The build emits a deterministic tar.gz, a companion manifest with each member's SHA256/size and the
archive SHA256/size, and `SHA256SUMS-v1`. It verifies the archive by streaming every member without
extraction or execution. This is local custody verification; root must separately upload and verify
public bytes. Additional support trees/files must be explicitly selected with the documented CLI
options. The tiny synthetic tests do not inspect real experiment payloads.

Only this new report directory is being edited. After root selects the final tracked paths, the
repository's generated data and entrypoint indexes must be regenerated as required by `agent.md` and
`DATA_INDEX.md`.
