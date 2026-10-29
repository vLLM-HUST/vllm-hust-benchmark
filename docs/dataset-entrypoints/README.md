# Dataset Entry Directory

Canonical, metadata-only source for the public dataset quickstart directory. This does not register
a new dataset executor or authorize a model run.

`scripts/build_dataset_entrypoints.py` reads the existing delivery catalogue, registered download
plans and per-source delivery ledger. `primary-clients.json` overlays the independently published
clients and sandbox workflows which are not represented by the preparation provider's entrypoint
list.

```sh
python scripts/build_dataset_entrypoints.py
python scripts/build_dataset_entrypoints.py --check
```

The output has 118 sources, 40 task classes, 15 separately classified assets, 104 provider
registrations and 14 provider-unregistered entries. Unknown provider registration is not evidence
that a project client is missing. No download-plan command is offered for those 14 entries.

The website mirrors the output at `assets/data/dataset-entrypoints-v1.json`. This separate
navigation asset does not replace the website's pinned experimental publication data. Its fixed
canonical commit and output SHA are recorded alongside the website copy.

Primary client commands were statically reviewed against their fixed sources; three core client
installations and 19 help commands were checked independently. These checks do not validate data
download, optional preparation dependencies, model requests or grading. Coder users follow the same
dataset-specific workspace instructions; the directory provides installation and execution guidance.

Ordinary sources share preparation, not an invented universal scorer. The existing request replay
module requires original `request_replay` bodies and is not an executor for static tasks or
metadata-only traces. Existing source-specific scripts retain their own environments and measured
scope. In particular, Terminal-Bench 2.1 requires an actual task-specific Harbor job; no universal
ordinary-user ARM64 job/bootstrap is supplied here.
