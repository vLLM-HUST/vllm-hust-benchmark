# Fixed proxy deployment against the retained model

RIDE one-click deployment created a new management Job and two fixed HTTP proxy Jobs. The original
model Job remained RUNNING and Ready; no model restart, cache-budget change or NPU release occurred.
The 42 planned payload files matched their readback hashes.

A separate named Ray Job sent one 16-token synthetic request through each new proxy, released each
program, and checked that engine running/waiting counts were zero before and after. This verifies
real request connectivity; it is not a real Agent batch, full RIDE lifecycle acceptance, or
performance result.

The preceding command-generation SyntaxError happened before any remote statement executed and is
retained as a failed deployment attempt. The earlier 256-Agent performance result remains invalid.
Its initial messages had only 8 distinct hashes, each repeated 32 times per arm; that property alone
does not prove why physical KV stayed low.

The source is aff94a7, already frozen in the sibling cancellation capsule. Private
source/plan/runtime/log evidence hashes are in report.json; raw task content, addresses and
credentials are not published. Run `python verify.py` for public consistency only.
