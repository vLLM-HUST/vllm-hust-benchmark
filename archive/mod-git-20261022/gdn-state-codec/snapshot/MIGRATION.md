# GDN State Codec 0.2 identity migration

GDN State Codec is an organization-owned recurrent-state codec, not a direct
replica of the paper mechanism that originally motivated the experiments.
Version 0.2 therefore uses one canonical identity throughout active code:

- MOD ID: `org.vllm-hust.gdn-state-codec`
- repository: `vLLM-HUST/gdn-state-codec`
- source/API stem: `gdn_state_codec`
- C++ type stem: `GdnStateCodec`
- native symbol stem: `statecentric_gdn_state_codec`

The retired-to-current mapping is:

| Retired identity | Current identity |
| --- | --- |
| `org.stateaxis.leapquant` | `org.vllm-hust.gdn-state-codec` |
| `leapquant` | `gdn_state_codec` |
| `LeapQuant` | `GdnStateCodec` |
| `statecentric_leapquant` | `statecentric_gdn_state_codec` |

Version 0.2 does not export runtime aliases for the retired identities.
Integrations must update the MOD ID, Python imports, CMake targets,
shared-library names, native symbols, and configuration keys together.

Historical evidence, rejected experiments, and archived patches retain their
original terminology and hashes. They describe the exact code that was tested
at the time and must not be rewritten as if it had used the 0.2 identity.
