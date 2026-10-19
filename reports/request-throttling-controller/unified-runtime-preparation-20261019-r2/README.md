# Frozen HUST runtime preparation and official bootstrap

The frozen installer and delivery manifest from main8c8e91adf588eef3e6d6ac4e2f79ae297f6529cf
prepared a new private directory. HUST Core/Ascend tags were resolved to the exact manifest commits,
both release assets downloaded through an authorized GitHub account, SDK extracted and all4695 files
independently rehashed. No upstream same-version pip package substitution.

The unchanged official healthy_runtime_entry.py check exits0 in the existing owned NPU-visible
worker. Loaded module paths/hashes and central configuration validation pass. No model was started
or migrated, no existing dependencies or test parameters changed, and the original Qwen service
remains healthy. This is source/bootstrap qualification only, not graph execution, quality or MOD
performance evidence. Package version labels describe the retained build; exact HUST commits and
loaded source bytes are authoritative.

The earlier zero-NPU Coder check failed during actual Triton Ascend architecture lookup (get_arch).
That negative remains in the separate V1 attempt capsule; no fake architecture, source patch or
dependency change bypassed it. This later attempt changes directory/device visibility only and uses
the same frozen inputs and official check arguments.

The private verification receipt is hash-bound in report.json. Public verify.py checks this capsule,
not the private raw receipt. Tokens, endpoints, host identities and full runtime environment remain
private; the one-use authentication file was deleted.
