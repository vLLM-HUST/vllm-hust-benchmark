# Targeted MOD original custody

This collection continues [issue 326](https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/326).
It preserves the KVCompress bilingual early-version result report and Extension Manager's
process-supervision acceptance report/companion record. KVCompress's source license is included;
Extension Manager has no tracked LICENSE file in this source tree (its package declares Apache-2.0).
Each manifest binds the public source repository, full source commit, Git blob, length and SHA256.
Failures, diagnostic/eager identity, original version labels and claim limits are unchanged. Custody
is not new hardware acceptance or a serving-performance claim.

- [KVCompress original manifest](compress/manifest.json)
- [Extension Manager original manifest](manager/manifest.json)
- [PyramidKV remaining current originals](pyramid-current/manifest.json): 98 records plus LICENSE
  and NOTICE, frozen from source commit `534bcc0` before consumer edits. Together with the existing
  paired capsule these preserve all 118 current evidence files. Original SHA256SUMS and their
  covered README files are retained byte-for-byte, rather than re-signed after link edits.
- Existing [PyramidKV paired originals](../mod-git-20261022/pyramidkv-website-source/manifest.json)
  and [vSpec originals](../vspec-browser-raw-20261022/manifest.json) are reused rather than uploaded
  again.

Source copies may be removed only after remote fixed-commit readback and per-file length/SHA256
verification, and after updating their consumers. Other historical collections, missing local inputs
and installation distributions are separate queues, not declared migrated by these capsules.

Skill review: E04/E31/E37/E42/E43/E47. No new measurement, host-volume write, raw-byte rewrite,
installation-asset deletion or gate bypass is authorized by the index. This batch uses the existing
persistent root volume with sparse checkout, not the ephemeral host workspace.
