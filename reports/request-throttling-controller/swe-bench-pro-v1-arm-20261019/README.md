# SWE-bench Pro V1 ARM: actual bounded Agent attempt

One real Qwen → unchanged official Agent → community ARM container → partial patch → fresh-container
official grading workflow executed. The Ray wrapper succeeded, but the Agent exhausted its frozen
30-call budget without submitting, and the official result is false. This is an unsuccessful repair
attempt, not a quality leaderboard or MOD performance result.

The Agent ran for296.066431s, produced a10015-byte partial diff across eight tracked files, and the
unchanged V1 grader ran for37.574659s in a separate fresh container. Whole attempt334.242417s. API
usage sums to411016 input/8308 output tokens over30 responses, all finish_reason=stop. Input counts
include repeated conversational history. Dollar cost was not measured.

The patch directly exports a function but consumers call its `.getOrCreate()` property. Official
test loading fails with `TypeError: require(...).getOrCreate is not a function`. The parser emits
zero test records, not681 individually failed tests. This failure is distinct from the retained
four-name mismatch in the earlier reference-patch check. The generated untracked self-check script
is retained in private trajectory evidence; the frozen partial-diff extraction includes tracked
files only. No extra model retries or human patch repair.

Private archive SHA256 and complete independent verification results are bound in report.json.239
file hashes,179 Agent files and9 official evaluator files checked; exact frozen input/task
formatting and loaded source identity verified.30 saved API usages recomputed; unchanged official
parser reproduces output.json byte-for-byte and official false. Docker events show distinct
Agent/grader containers; no task containers remain. Public verify.py checks capsule consistency
only, not the private archive.

Pro V1 is sufficient for the engine-performance objective. Reuse ready ARM images and official
tools; V2, x86 scoring, a successful model repair or full quality campaign are not prerequisites.
Next freeze the Native/Controller batch contract independently; do not relabel this single-task
elapsed time as scheduling benefit. Full raw prompts, model responses, patches, test payloads,
endpoints and credentials remain private.

Source:
[official V1 tools](https://github.com/scaleapi/SWE-bench_Pro-os/tree/66f92766bba642462d4bbe5479e83f91f9211862),
[official Agent](https://github.com/scaleapi/mini-swe-agent/tree/d74716a3c8104a113f77cc9ab94cf407ecdcf1e9),
[community ARM recipes](https://github.com/bdqnghi/swe_benchmark_arm/tree/b2834f1b0cf94294f409e93c7dccc398b0a2699b).
