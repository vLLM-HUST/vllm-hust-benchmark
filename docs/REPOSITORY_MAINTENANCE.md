# Repository maintenance

## Small source workspaces

Use a local code directory outside cloud document synchronization. For code and publication work,
partial clone and sparse checkout avoid downloading historical evidence on first checkout.

```sh
git -c core.longpaths=true -c core.autocrlf=false clone --depth 1 --filter=blob:none --sparse https://github.com/vLLM-HUST/vllm-hust-benchmark.git
cd vllm-hust-benchmark
git sparse-checkout set .github .agents src scripts tests docs schemas leaderboard-data website-publication
```

Add the exact report or archive collection needed for a task with
`git sparse-checkout add reports/<collection>`. Add `workload-preparation` when maintaining the
independent data package. Tests which read historical evidence require those collections too. Sparse
checkout changes which files are materialized locally. The tracked Git index still records the
complete repository for navigation generation.

For data-package installation, use
[the pinned wheel installation guide](workload-data/INSTALLATION.zh-CN.md). The Git tree retains
fixed evidence paths and byte identities. A full evidence checkout requires sufficient disk space
and Windows long-path support.

## Navigation maintenance

Stage intended file additions and removals first, then refresh the existing navigation tools in
their dependency order.

```sh
git add <intended-paths>
python scripts/refresh_data_navigation.py --write
python scripts/refresh_data_navigation.py --check
```

The command checks the collection index, website/Release entrypoints and published evidence links.
The last generator reads evidence extracts. Materialize the referenced collections before running
that generator. A missing sparse file is a checkout limitation and must be resolved by fetching the
source collection.

Review the generated diffs, stage the navigation files and run the same check again. Remote Release
inventory refresh remains an explicit `data_entrypoints.py --refresh` operation with a reviewed
website revision, as described in [DATA_INDEX.md](../DATA_INDEX.md). CI performs the offline checks
against that recorded inventory.

## Publication and historical files

Website data originates in [website-publication](../website-publication/README.md). Edit the
benchmark source, validate it, merge it, then update the reviewed website publication revision.
Legacy leaderboard exports remain documented in the historical wrapper examples in the root README.
The scheduled website preview and production Pages have different inputs.

New large raw archives should use a versioned Release attachment with a committed manifest that
records the asset URL, exact SHA256, byte count and member paths. Keep small summaries, download
instructions and verification entrypoints in Git. Before relocating an existing archive, verify its
consumers and exact bytes at the replacement location. Historical evidence updates append a new
version with a supersedes relationship.

Website root slides and already tracked browser output need the same ownership and consumer audit
before migration. A file move alone cannot reduce the size of earlier Git history. Partial checkout
provides a bounded source workspace while existing download and reproduction links are retained.
