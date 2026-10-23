# Real TP2 worker invocation gate qualification

This retained campaign ran Qwen2.5-7B-Instruct on physical Ascend 910B2 devices
3 and 6, exposed as logical devices 0 and 1 with ASCEND_RT_VISIBLE_DEVICES unset.
Manager source was merged main 484d6fd16180065784729f23d2abd85f229f5049;
Host was the unmerged 379a0ad35 candidate wheel, Ascend was 873584026.
The input audit checked actual wheel hashes, every installed Python source
against its wheel, and all 14 model files. The Plan used the actual Host version
and kv-materialization wheel digest. Inputs were frozen before startup.

The independent HTTP authority consumed real Host loader invocation journals
and rechecked live process identities. With worker 0's journal renamed intact
into quarantine, candidate open was refused and HTTP returned the closed
predecessor's 503. Restoring the exact bytes allowed a real completion through
port 8187: HTTP 200, `ECPA-GATE-OK`, 8 generated tokens. Renaming the same journal
again reduced invocation coverage from 1.0 to 0.5, revoked the candidate route,
and returned 503. Both blocked request boundaries left the backend log bytes
unchanged, while direct backend health remained 200. The authority journal
binds the sole candidate admission to the exact observation and records the
post-revocation rejection. All owned processes, ports, and NPUs were cleaned.

Classification is **QUALIFIED_NONFORMAL**. This establishes this one real worker
invocation coverage mechanism and independent HTTP routing behavior. It does
not establish worker effective semantics, H1 prevention, formal #60/H3/H4 starts,
model quality, output correctness, rollback, or performance. The Host publish
and merged-main admission gate remains unsatisfied.

`retained.tar.gz` preserves exact raw inputs, Host journals, Linux stat/cmdline
snapshots, quarantine bytes, independent authority trace, HTTP responses,
backend logs, cleanup, and execution/source snapshots. It also retains four
failed/aborted attempts and their original results/logs; the failure index keeps
the original hashes and corrected failure classifications. These attempts are
not qualification evidence. `manifest.json` binds every retained raw member and
the compressed archive. Verify without launching a model:

```sh
PYTHONPATH=src .venv/bin/python -m experiments.exposure_gate.qualification_verify \
  --package experiments/exposure_gate/artifacts/real-invocation-gate-qualification-20261013
```

The verifier checks raw journal line bindings, actual worker invocation coverage,
process identities and cmdlines, exact authorization and snapshot bindings,
the authority hash chain and deny/open/revoke ordering, real inference, and
cleanup. It never promotes this package into a formal registration.
