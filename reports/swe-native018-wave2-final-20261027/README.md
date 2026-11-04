# SWE Native 0.18: Wave 2 Final Three Tasks

This directory adds Protonmail, qutebrowser and Tutanota original single-attempt results to the
unchanged 26-task catalog. The resulting scope is 29 of 731 tasks: 4 original official positives, 5
positives after the previously disclosed same-answer Protonmail dependency recovery, and 702
unattempted tasks.

Protonmail's original official false is retained. Ten ConversationView tests failed at runtime; the
selected hook reported no tests found. The four parser entries are file-level fallback placeholders,
not four target assertions. qutebrowser passed all ten required tests. The original trailing XIO
message after the ten passes is retained without rerunning grading. Tutanota completed its type
build but Esbuild could not load esbuild-linux-arm64. Its 107 missing targets are not 107 assertion
failures.

Run `python3 verify.py` to rebuild and verify the catalog, source hashes, independent public review
projections and unchanged recovery denominator. Private task, answer, patch and trajectory contents
are not included. Derivation source copies identify the original offline processing, not new
executions.
