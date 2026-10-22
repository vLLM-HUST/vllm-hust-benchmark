# Probe search: promisor completion and nested archives

The [receipt](search-receipt.json) records two additional searches. All 2,872 missing objects in the
local benchmark ref/reflog traversal were fetched; 10,753 local blobs were hashed and JSON
provenance mappings checked, without recovery. An older mounted benchmark clone contributed another
1,264 checked blobs. The eight previously verified Release archives contained 34 recognized nested
containers and 14,607 regular nested members; no target identity or digest matched.

All 33 unresolved occurrences remain unresolved. This does not prove permanent loss. Offline disks
cannot be accessed through the current container device namespace; the configured storage host
rejected SSH authentication. No mounting, filesystem repair, credential changes, raw-device
recovery, or hardware experiments were performed. Unknown machines, deleted/pruned remote objects
and unsupported archive formats remain outside the verified search scope.

## Bare-metal agent handoff

Use this task on an authorized bare-metal machine. Its original role need not be known: first
discover available volumes and old workspaces, then search.

1. Work read-only against sources. Do not rerun experiments, rewrite logs, clean disks, run Git
   GC/prune, start old containers, or execute archived programs.
1. Download the fixed
   [reference audit](https://raw.githubusercontent.com/vLLM-HUST/vllm-hust-benchmark/6864562eb2feb741a904ca18822cc2481983e039/archive/mod-git-20261022/split-probe-recovery/local-reference-audit.json)
   into a new temporary directory. Verify SHA256
   `0501afbd761229778ca66efc986cf68164262908860f39d34858bfad1ae60554`. Target only the 33 rows where
   `canonical_raw_url` is null. Preserve each row's original path, source document, location and
   declared digest.
1. Inspect machine identity, existing mounts, old container volumes and backup inventories. Look for
   copies of `/root/包体恢复/records/probes/` and `/root/prof/` in old root filesystems and user
   workspaces. Priority names: `20261019T034200Z_gdn_registration`, `20261019T042000Z_opt_repair`,
   `20261018T164700Z_gdn_separate`, `ladder-1019-021225`, and `ladder-1019-022857`. Include the
   audit's remaining workflow/operator logs.
1. For unmounted volumes or inaccessible hosts, report exact targets and request owner
   authorization. Do not mount, change permissions/trust, repair filesystems, scan arbitrary hosts,
   or attempt undelete/raw-device recovery on your own.
1. Search historical Git objects and nested archives without filesystem extraction or execution.
   Record unavailable objects, encrypted archives and resource limits. Do not inspect credentials,
   tokens, or unrelated private user data.
1. For every candidate, report source machine/volume privately, complete path or archive-member
   chain, byte length, SHA256, and original-path/provenance mapping. A known digest must match
   exactly. Without a digest, require independent origin evidence; a filename or report assertion
   alone is insufficient. A directory reference needs a complete member inventory, not a partial
   file recovery.
1. Return a candidate manifest, search coverage and blockers first; do not upload publicly. After
   review, copy confirmed candidates to a fresh staging directory and hash both source and copy
   again. Preserve original bytes and failed records. Missing originals stay unavailable; new runs
   are not historical replacements.
