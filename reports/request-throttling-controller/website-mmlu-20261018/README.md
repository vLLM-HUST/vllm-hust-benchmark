# MMLU-Pro website publication evidence

PR445 publishes Controller9599/12032 (79.778923%) in the original MMLU row and keeps historical
Native9609/12032 unchanged. Matching Native9632/12032 appears only in the existing dated
seven-column details. Both full quality passes use5shot/max2048/client16;66/62 invalid answers
and1536/1520 truncations remain, with zero transport failures. Quality retention99.657392% passes
the99% budget, without statistical significance or automatic cost qualification.

Independent performance uses200 prompts, outputcap256/naturalEOS/temp0/seed0/rateinf/no client cap:
one warmup plus three formal runs per arm. All eight runs completed200/200 with zero failures.
Formal median output337.200931/360.525152 tokens/s (+6.917010%); this ordinary single-turn
forwarding test does not establish program scheduling causality or full-solution cost. Aggregate
TTFT/TPOT cannot be independently reconstructed from per-request timings that were not retained.

Four exact-head checks passed. CI screenshots at1440/390/320 were inspected. The inherited
screenshot names use language suffixes that differ from their final toggle state: detail-390-en.png
actually shows Chinese; detail-320-zh.png shows English. Actual Chinese and English published pages
were separately inspected in the Codex in-app browser, including the overview, all six MMLU records,
and expanded runtime/audit evidence. Live screenshot and accessibility transcripts are included. No
live Chrome inspection is claimed. Pages built the exact merge; JavaScript, CSS, generated snapshot
and MMLU source report matched checkout bytes. HTML matched after removing exactly one Cloudflare
analytics script line, explicitly recorded in the receipt. Revision evaluation remains incomplete.
