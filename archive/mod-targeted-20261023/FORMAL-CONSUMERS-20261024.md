# Formal archives and mixed-document consumers

Custody continuation of #326, observed 2026-10-24 (Asia/Shanghai), not new hardware validation.

| Source                                                       | Frozen scope                                                         | Consumer disposition                                                                                                                                                       |
| ------------------------------------------------------------ | -------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Extension Manager `ef4662976adc02ca86739dc4bd0f56e5e7e35511` | [77 members](manager-formal/manifest.json), including README context | formal_campaign and native_challenge originals; frozen portable verifier bytes stay unchanged; callers use checked external inputs                                         |
| KVCompress `c077a6a0f4854e065d68bef52f742b71bd654536`        | [5 members](compress-mixed/manifest.json)                            | paired English/Chinese optimization and method documents plus license; current implementation stays in source, historical measurements move to these byte-preserved copies |
| vSpec `1b6a061f9ab86295d8f1a305874a4a9e3ba227ab`             | [5 members](vspec-mixed/manifest.json)                               | migration_status and three dated validation/recovery documents plus license; old paths retain navigation to these originals                                                |

Source manifests record the exact Git blobs, lengths and SHA256. Existing archive parts are copied
without repacking, re-signing or interpreting failures as passing results. Their original internal
manifests and portable-verifier bindings remain authoritative. The directory name `formal_campaign`
does not imply formal admission: partial, diagnostic, failed and nonformal records retain their
scope. CPU challenge attempts do not imply NPU qualification. Historical B0/B1 labels inside
immutable documents are not definitions of the current project's fixed 0.18/0.18 B0.

Some archived Markdown uses original relative links and local paths. Those record the source
checkout's context, not a claim that the referenced local original has been recovered. Fixed source
commits in the manifests can resolve implementation references; previous benchmark manifests resolve
already-migrated result references. Current source navigation links to these frozen documents.

Remaining combination-experiment generators and mixed input/output consumers require classification;
archiving an input alone does not make every consumer safe to delete. No installer asset is deleted
without verified consumer closure. Triton review runner, LMCache device checks, SGLang approvals and
bounded missing Profiler/KV inputs retain their existing owners and blockers. The 33 old probes are
not pursued; cancelled GDN retesting remains cancelled.

Review: requested migration | source/manifests and consumer tests | E04/E31/E34/E37/E42/E43/E47 |
preserve raw bytes, failure identities, frozen verifiers and synthetic fixtures; persistent storage
only.
