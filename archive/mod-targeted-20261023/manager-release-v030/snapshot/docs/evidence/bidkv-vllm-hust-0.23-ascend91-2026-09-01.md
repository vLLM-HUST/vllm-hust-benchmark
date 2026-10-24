# BidKV typed scheduler policy online gate on Ascend 91

Date: 2026-09-01  
Host: `ascend91-host`, one idle Ascend 910B2  
Result: online policy selection and next-process rollback passed

## Pinned inputs

- vLLM-HUST core commit: `5c994cdc0` (typed scheduler policy host seam)
- Extension Manager commit: `e71d7cf`
- BidKV commit: `d0a7b20`
- container image:
  `local/vllm-ascend-hust@sha256:f86a810132b152f93579ad993eb8d1b3202fd901d664b1dc468385d18acc313f`
- installed distributions: vLLM `0.23.0+empty`, vLLM Ascend `0.23.0rc1`,
  `vllm-hust-ext 0.2.0.dev0`, and BidKV `0.1.1`
- model: local Qwen3-0.6B checkpoint
- service: loopback-only `127.0.0.1:18091`

The existing 0.23 image predates the current scheduler file. The test copied
the new typed admission modules and victim-selector implementation into an
isolated container, then applied only the generic selector construction and
preemption-selection calls to that image's native scheduler. It did not copy
the newer scheduler wholesale or modify any persistent host installation.

## Enabled path

Manager discovered BidKV, projected
`installed + discovered + compatible + configured + enabled`, rendered a
host-native schema 1.0 manifest, and selected
`org.vllm-hust.bidkv/victim-selector`. EngineCore reported:

```text
Loaded typed victim selector component=org.vllm-hust.bidkv/victim-selector
source=.../bidkv/adapters/vllm_hust/selector.py api_version=1
```

With `max_model_len=2048` and a 2,560-token KV cache, three concurrent
1,400-token generations forced real KV pressure. BidKV was called three times
at `kv_util=1.00`; representative events selected victims from three and two
running requests:

```text
[BidKV] UTILITY_ACTIVE ... r=769 tok ... running=3
[BidKV] UTILITY_ACTIVE ... r=1152 tok ... running=2
[BidKV] UTILITY_ACTIVE ... r=1407 tok ... preemptions=1 ... running=2
```

All three requests completed with 1,422 total tokens and `finish_reason=length`.

## Disable and next-process rollback

After `extension disable`, Manager preserved installed/discovered/compatible/
configured state but removed enabled intent. The next Manager dry run contained
no native extension manifests and no additional selector configuration. The
replacement service started as `builtin-qwen`, returned a successful real
completion, and its complete log contained zero typed-selector load lines and
zero `[BidKV]` lines. The process environment had neither
`VLLM_EXTENSION_MANIFESTS` nor `VLLM_EXTENSION_BUNDLES`.

Raw logs retained by the task workspace:

- enabled log SHA-256:
  `cbb8c86438633e7c48ad66fdfdc06a970f89f0a4f48bea31835bfcdb8925b4c9`
- built-in rollback log SHA-256:
  `aeeeebf7d924b936103a5a6658d219318d2e5fd0a32d5b8c1fe2632c04b5dc38`

The isolated container and remote staging directory were removed. A final
`npu-smi` check reported no processes on all eight devices.

The run also exposed a non-fatal naming warning: the Manager diagnostic
variable used the reserved-looking `VLLM_` prefix. It was renamed to
`VLLMHUST_EXT_ENABLED_BUNDLES` after this run so future launches do not present
it as an unknown vLLM-owned setting.

## Remaining release boundary

The first isolated run closed the online load, policy-call,
request-completion, disable, and next-process fallback behavior gate. The
following clean-artifact repeat closes that remaining 91 packaging gap; the
cross-host release matrix is still open.

## Clean pushed-artifact repeat

The repeat used only artifacts derived from pushed commits:

- vLLM-HUST `87096bd3de782b2cd32d0dfd4efcbc891adad268`;
- BidKV `2b55997`, wheel SHA-256
  `cb10f1a8fbfb5a9ec8f64bb163baddf7719a0071618836ad3804a05823444f27`;
- Extension Manager `b4f221f`, wheel SHA-256
  `6e467fa276cfa64a1cfff7655e38ca11d654dc7aa111526bef7e8dfc2e726123`;
- source archive SHA-256
  `0888a19706641c3539cf971ed9f639bfa41debd01a90562f5ba4f27bd9e6c1d0`;
- resulting arm64 image manifest
  `sha256:81e3dfba8affd27c6ace86c39a026d4383a1a908d80ca24e0226d5bfb0b42ee3`.

The clean carrier initially exposed a packaging defect: copying only
`victim_selector.py` admitted BidKV but did not invoke it. Copying the entire
fork scheduler was then rejected because it drifted from the pinned Ascend
base. The final carrier instead applies four fail-closed edits to the native
base scheduler: import, construction, preemption selection, and observability.
The build aborts if any pinned fragment changes.

With the corrected carrier, EngineCore loaded the typed component and three
concurrent 1,400-token generations produced three real selections at
`kv_util=1.00` (`r=769`, `r=1152`, and `r=1407`). All requests ended with
`finish_reason=length` and 1,400 completion tokens.

After Manager disable, the next process used the same carrier without the
BidKV wheel or extension environment. It completed a real 64-token request;
its full log contained zero `Loaded typed victim selector` and zero `[BidKV]`
matches. Manager `forget` produced an empty extension map. In a fresh container
virtual environment, uninstalling BidKV removed its
`vllm_hust.extension_bundles` entry point and made `import bidkv` fail; the main
wheel never registered a BidKV `vllm.general_plugins` entry point. The temporary
services were removed and NPU 5 had no remaining process.

This passes the clean install, enable, real invocation, disable, fallback,
forget, and uninstall gate on server 91.

## Clean cross-host repeat on 112

The exact same source archive and wheel hashes were transferred to arm64
server 112 and rebuilt from the same pinned official Ascend base. The resulting
local image ID was
`sha256:77b159e0c51a89ba75cca289beb0dd56e68f5f4eec400f280979b4121ae1dc43`.
No source checkout or system Python package was modified.

Qwen2.5-3B-Instruct started on one isolated Ascend 910B2. EngineCore loaded the
same typed component. Three concurrent 1,400-token generations again produced
three real `UTILITY_ACTIVE` selections at `kv_util=1.00` (`r=769`, `r=1152`,
and `r=1407`); all three ended with `finish_reason=length` and 1,400 completion
tokens. After disable, the next built-in process completed a 64-token request
and its full log contained zero typed-selector or BidKV matches. Forget yielded
an empty extension map. A clean virtual-environment uninstall removed the
Bundle entry point and both Python imports.

Both test containers and temporary installation directories were removed;
NPU 7 reported no remaining process. The retained wheel/source artifacts are
inputs, not a runner or maintained service. This completes the clean 91/112
packaging and online-repeat gate. Alpha remains blocked on the broader version,
permission and failure matrix; these results do not expand support to official
vLLM.
