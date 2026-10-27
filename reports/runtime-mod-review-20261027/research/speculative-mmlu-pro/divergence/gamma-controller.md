# vSpec gamma controller: algorithm versus observed35B behavior

Source: vSpec4e17b6bb6f9957dd4427f112da0bfc3d08bf24ae,
`src/vllm_hust_vspec/adaptive/{online,runtime}.py`. Actual configuration:
`../q35-environments.json`. Historical runtime summaries extracted verbatim into
`historical-gamma-summaries.json` from four formal35B logs (quality280 and performance56 r1/r2/r3).

## Critical correction

All four formal runs report `candidate_gammas=[2,4,6]`, `initial_gamma=4`, `gamma_switches=0`,
`selection_counts={'4':3}`, and only `online_ucb_warmup_window` selection reasons. Accepted-position
observations for positions5/6 are0. Therefore the adaptive configuration was enabled, but **these
runs did not demonstrate dynamic gamma switching**. Observed behavior stayed atgamma4. Earlier
report labels “2/4/6+refill” describe configuration only and must not imply exercised switching. The
whole-stack speed measurements remain observations, but compare actualgamma4 candidate versus
nativegamma2, not isolated online adaptation.

## What the online controller is intended to optimize

- Gamma is draft proposal depth per request; target verifies up togamma+1 positions including the
  bonus/fallback token. It does not set an acceptance-relaxation threshold.
- `runtime._online_initial_gamma` uses a hand-chosen MTP anchor4, clamped to configured range. MTP
  candidate_stride2 gives2/4/6 here. Starting at4 is a prior, not an online discovery.
- `OnlineGammaController.batch_bucket` groups concurrency by next power of two (B9–16 shareB16
  statistics). These are **statistics buckets**, not graph capture buckets.
- `observe` collects accepted drafts and useful tokens (`accepted+1` per request), position
  acceptance and a request-level second-token continuation statistic. Acceptance uses EWMA0.1 here.
- `complete_step` records useful tokens / elapsed milliseconds. Runtime uses wall-time between
  scheduler output completions, not isolated NPU kernel timing. Only stable decode steps (no
  scheduled new requests and matching query width) contribute reward; other steps discard reward but
  retain acceptance feedback.
- Each gamma/batch bucket keeps a32-sample reward window. Reward is sum(useful tokens)/sum(latency),
  not unweighted mean of per-step rates.
- Unmeasured-arm prediction estimates `E(useful)=1+sum_j product_{i<=j} p_i`; positions with
  insufficient evidence use a continuation estimate. Serial cost scales asgamma+2 for MTP (shared
  target-cost prior2), relative to measured anchor.
- Probe expansion is model-gated: anchor needs32 observations and predicted improvement over best
  measured arm must exceedmax(hysteresis,exploration),3% here. It does not blindly benchmark every
  arm each interval.
- Generic selection is UCB-like, but **online exploration coefficient is0.0 in our configuration**:
  the uncertainty bonus is disabled. This does not disable initial/model-gated probing.
- Control interval4, warmup samples2 (+1 async scheduling padding), hysteresis3%, and
  max_gamma_step2 limit switching. “Interval4” means controller selection/residence opportunities,
  not an unconditional change every four device steps.
- Online mode ignores `context_tokens` in reward updates and passes0 from scheduler. It does not
  directly estimate prompt semantics, difficulty or per-request entropy here; entropy stopping is
  off.

## Why no switching occurred here

`runtime_switch_env` is `HUST_VSPEC_MTP_ASYNC_RUNTIME_SWITCH`, default0 and absent from frozen
formal environments. `_should_hold_eagle3_async_cohort` also applies toMTP: with async and active
running requests, it holds the current width unless runtime switching is enabled. The native
cohort-refill fast path can return without invoking `choose`. Width changes remain eligible at
empty-cohort boundaries, preserving recurrent/async state layout safety. The actual logs show only
three selection counts atgamma4 and no transitions, so this workload never exercised online depth
adaptation.

Do not silently enable this switch in the existing protocol: it changes state-layout/runtime
behavior and needs a separate correctness admission. Do not attribute the no-switch observation to
the short synchronous probe; it is independently present in all four uninstrumented formal logs.

## Source anchors

- `online.py`: `_ArmWindow.mean_reward`, `_expected_useful_tokens`, `_modeled_reward`, `observe`,
  `complete_step`, `choose`.
- `runtime.py`: `_online_probe_shared_cost_units`, `_online_initial_gamma`,
  `_should_hold_eagle3_async_cohort`, `apply_adaptive_patches` (`runtime_switch_env`, scheduler
  selection/hold, completion feedback).
