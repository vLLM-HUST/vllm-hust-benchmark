# First256-Agent NC pair: mechanism observed, performance invalid

Both arms actually ran256 independent V1 ARM Agent instances (same eight tasks×32), time0, without
client slots. Frozen HUST model/KV/profile and Agent budgets were unchanged. The Ray job finished
normally, but each arm has256/256 RetryError with inner model Timeout, zero Submitted. All512 grades
remain unrun/unresolved, not0% scores.

| Observed item                                     |      Native |  Controller |
| ------------------------------------------------- | ----------: | ----------: |
| Time until every Agent exited with error, seconds | 1138.783284 | 1203.675442 |
| Saved response calls                              |        1862 |        2045 |
| Saved response output tokens                      |      109203 |      114602 |
| Real KV occupancy peak                            |  15.594059% |  15.965347% |
| Engine preemptions                                |           0 |           0 |
| Logged actual pause / resume events               |       0 / 0 |   418 / 406 |
| Unresolved grading                                |         256 |         256 |

Controller also logged472 pause markings, which are not counted as actual pauses. Actual
paused/resumed distinct programs are252/244. The mechanism occurred without physical KV saturation:
no sampled occupancy reached90%. Prefix hit-token fractions were58.77%/60.25%. These observations do
not prove that prefix reuse alone caused low occupancy.

The pair is invalid for performance: all Agents timed out and work differs. Native exited with 15
running and27 waiting downstream requests; Controller started with16 running and25 waiting requests
and15.10% KV. A program-release receipt is not proof of downstream request cancellation. Do not
interpret the+5.698376% error-exit time difference as a valid performance result. No further CN/NC
repetition was started. A new entry gate requires both running/waiting counts zero before starting
any new Agent batch; missing metrics or timeout rejects the launch. The follow-up gate has16 passing
entry tests and was not applied to the historical run.

The private44,846,170byte archive SHA is in report.json. Independent audit checked7603 file
hashes,512 unique containers/programs, arrivals, each saved response usage, batch summaries,
error-exit duration and release receipts. Model/proxy logs and runtime receipts are separate private
evidence; they are not claimed to be inside that CPU archive. Snapshot sources distinguish measured
code from follow-up fixes. Public verify.py checks public consistency only.

PNG/PDF figures reuse existing plot_run and plot_swe_bench_pro layouts, palette and10second rate
convention. Failed-batch plot wording explicitly identifies error-exit time and saved response
tokens. KV token axes are engine-reported hybrid-capacity equivalents. Graphs are not per-request
latency reconstruction. Real Grafana/Ray screenshots and log attachments are in the original Feishu
report. Start/terminal screenshots exist, but some intermediate10minute screenshots are missing;
full per-second observations remain. Both owned cards are retained.
