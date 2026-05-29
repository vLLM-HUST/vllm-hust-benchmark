#!/bin/bash
set -euo pipefail

SOURCE_BENCHMARK_REPO_DIR=${SOURCE_BENCHMARK_REPO_DIR:?SOURCE_BENCHMARK_REPO_DIR is required}
TARGET_BENCHMARK_REPO_DIR=${TARGET_BENCHMARK_REPO_DIR:?TARGET_BENCHMARK_REPO_DIR is required}
WEBSITE_REPO_DIR=${WEBSITE_REPO_DIR:?WEBSITE_REPO_DIR is required}
PYTHON_BIN=${PYTHON_BIN:-python3}
SNAPSHOT_TARGET_BRANCH=${SNAPSHOT_TARGET_BRANCH:-main}
SNAPSHOT_OUTPUT_DIR=${SNAPSHOT_OUTPUT_DIR:-$TARGET_BENCHMARK_REPO_DIR/leaderboard-data/snapshots}
LOCAL_SNAPSHOT_OUTPUT_DIR=${LOCAL_SNAPSHOT_OUTPUT_DIR:-}
SNAPSHOT_SOURCE_PATTERN=${SNAPSHOT_SOURCE_PATTERN:-official-ascend-*}
SNAPSHOT_MAX_PUSH_ATTEMPTS=${SNAPSHOT_MAX_PUSH_ATTEMPTS:-4}
SNAPSHOT_PUSH_RETRY_SECONDS=${SNAPSHOT_PUSH_RETRY_SECONDS:-5}
SNAPSHOT_COMMIT_MESSAGE=${SNAPSHOT_COMMIT_MESSAGE:-chore(data): publish official ascend baseline snapshots}
GIT_COMMITTER_NAME=${GIT_COMMITTER_NAME:-vLLM-HUST Benchmark Bot}
GIT_COMMITTER_EMAIL=${GIT_COMMITTER_EMAIL:-benchmark-bot@vllm-hust.local}
BENCHMARK_REPO_REMOTE=${BENCHMARK_REPO_REMOTE:-origin}
BENCHMARK_REPO_SLUG=${BENCHMARK_REPO_SLUG:-vLLM-HUST/vllm-hust-benchmark}
BENCHMARK_REPO_GH_TOKEN=${BENCHMARK_REPO_GH_TOKEN:-}
BENCHMARK_REPO_SSH_KEY=${BENCHMARK_REPO_SSH_KEY:-}

required_submission_files=(leaderboard_manifest.json run_leaderboard.json)
required_snapshot_files=(
  leaderboard_single.json
  leaderboard_multi.json
  leaderboard_compare.json
  last_updated.json
)

write_github_env() {
  local key=$1
  local value=$2
  if [[ -n "${GITHUB_ENV:-}" ]]; then
    printf '%s=%s\n' "$key" "$value" >>"$GITHUB_ENV"
  fi
}

configure_push_remote() {
  local remote_url=

  if [[ -n "$BENCHMARK_REPO_GH_TOKEN" ]]; then
    remote_url="https://x-access-token:${BENCHMARK_REPO_GH_TOKEN}@github.com/${BENCHMARK_REPO_SLUG}.git"
    git -C "$TARGET_BENCHMARK_REPO_DIR" remote set-url "$BENCHMARK_REPO_REMOTE" "$remote_url"
    return 0
  fi

  if [[ -n "$BENCHMARK_REPO_SSH_KEY" ]]; then
    remote_url="git@github.com:${BENCHMARK_REPO_SLUG}.git"
    git -C "$TARGET_BENCHMARK_REPO_DIR" remote set-url "$BENCHMARK_REPO_REMOTE" "$remote_url"
    return 0
  fi

  if [[ "${GITHUB_ACTIONS:-}" == "true" ]]; then
    echo "Either BENCHMARK_REPO_GH_TOKEN or BENCHMARK_REPO_SSH_KEY is required for direct benchmark publication in GitHub Actions" >&2
    exit 2
  fi

  return 0
}

if [[ ! -d "$SOURCE_BENCHMARK_REPO_DIR/submissions" ]]; then
  echo "source benchmark submissions directory not found: $SOURCE_BENCHMARK_REPO_DIR/submissions" >&2
  exit 2
fi

if [[ ! -d "$TARGET_BENCHMARK_REPO_DIR/.git" ]]; then
  echo "target benchmark repository checkout not found: $TARGET_BENCHMARK_REPO_DIR" >&2
  exit 2
fi

if [[ ! -f "$WEBSITE_REPO_DIR/scripts/aggregate_results.py" ]]; then
  echo "website aggregation script not found: $WEBSITE_REPO_DIR/scripts/aggregate_results.py" >&2
  exit 2
fi

if [[ "${GITHUB_ACTIONS:-}" != "true" && "${ALLOW_LOCAL_GIT_RESET:-0}" != "1" ]]; then
  echo "refusing to reset a local checkout outside GitHub Actions; set ALLOW_LOCAL_GIT_RESET=1 to override" >&2
  exit 2
fi

shopt -s nullglob
source_submission_dirs=("$SOURCE_BENCHMARK_REPO_DIR"/submissions/$SNAPSHOT_SOURCE_PATTERN)
shopt -u nullglob

if [[ ${#source_submission_dirs[@]} -eq 0 ]]; then
  echo "no source submissions matched pattern '$SNAPSHOT_SOURCE_PATTERN' under $SOURCE_BENCHMARK_REPO_DIR/submissions" >&2
  exit 2
fi

for source_submission_dir in "${source_submission_dirs[@]}"; do
  for file_name in "${required_submission_files[@]}"; do
    if [[ ! -f "$source_submission_dir/$file_name" ]]; then
      echo "missing source submission file: $source_submission_dir/$file_name" >&2
      exit 2
    fi
  done
done

relative_snapshot_dir="leaderboard-data/snapshots"

git -C "$TARGET_BENCHMARK_REPO_DIR" config user.name "$GIT_COMMITTER_NAME"
git -C "$TARGET_BENCHMARK_REPO_DIR" config user.email "$GIT_COMMITTER_EMAIL"
configure_push_remote

prepare_publication_commit() {
  local relative_submission_paths=()
  local source_submission_dir
  local run_id
  local target_submission_dir
  local file_name

  git -C "$TARGET_BENCHMARK_REPO_DIR" fetch "$BENCHMARK_REPO_REMOTE" "$SNAPSHOT_TARGET_BRANCH"
  git -C "$TARGET_BENCHMARK_REPO_DIR" checkout -B "$SNAPSHOT_TARGET_BRANCH" "$BENCHMARK_REPO_REMOTE/$SNAPSHOT_TARGET_BRANCH"

  mkdir -p "$TARGET_BENCHMARK_REPO_DIR/submissions"
  for source_submission_dir in "${source_submission_dirs[@]}"; do
    run_id=$(basename "$source_submission_dir")
    target_submission_dir="$TARGET_BENCHMARK_REPO_DIR/submissions/$run_id"
    relative_submission_paths+=("submissions/$run_id")

    rm -rf "$target_submission_dir"
    cp -a "$source_submission_dir" "$target_submission_dir"
  done

  mkdir -p "$SNAPSHOT_OUTPUT_DIR"
  for file_name in "${required_snapshot_files[@]}"; do
    rm -f "$SNAPSHOT_OUTPUT_DIR/$file_name"
  done

  "$PYTHON_BIN" "$WEBSITE_REPO_DIR/scripts/aggregate_results.py" \
    --source-dir "$TARGET_BENCHMARK_REPO_DIR/submissions" \
    --output-dir "$SNAPSHOT_OUTPUT_DIR"

  for file_name in "${required_snapshot_files[@]}"; do
    if [[ ! -f "$SNAPSHOT_OUTPUT_DIR/$file_name" ]]; then
      echo "missing generated snapshot file: $SNAPSHOT_OUTPUT_DIR/$file_name" >&2
      exit 2
    fi
  done

  if [[ -n "$LOCAL_SNAPSHOT_OUTPUT_DIR" ]]; then
    mkdir -p "$LOCAL_SNAPSHOT_OUTPUT_DIR"
    for file_name in "${required_snapshot_files[@]}"; do
      cp "$SNAPSHOT_OUTPUT_DIR/$file_name" "$LOCAL_SNAPSHOT_OUTPUT_DIR/$file_name"
    done
  fi

  git -C "$TARGET_BENCHMARK_REPO_DIR" add "${relative_submission_paths[@]}" "$relative_snapshot_dir"
  if git -C "$TARGET_BENCHMARK_REPO_DIR" diff --cached --quiet; then
    return 1
  fi

  git -C "$TARGET_BENCHMARK_REPO_DIR" commit -m "$SNAPSHOT_COMMIT_MESSAGE"
}

for attempt in $(seq 1 "$SNAPSHOT_MAX_PUSH_ATTEMPTS"); do
  if ! prepare_publication_commit; then
    echo "Official baseline publication is already up to date on ${BENCHMARK_REPO_SLUG}@${SNAPSHOT_TARGET_BRANCH}"
    write_github_env GITHUB_SNAPSHOT_SYNC_STATUS unchanged
    exit 0
  fi

  snapshot_commit=$(git -C "$TARGET_BENCHMARK_REPO_DIR" rev-parse HEAD)
  if git -C "$TARGET_BENCHMARK_REPO_DIR" push "$BENCHMARK_REPO_REMOTE" "HEAD:$SNAPSHOT_TARGET_BRANCH"; then
    echo "Pushed official baseline publication to ${BENCHMARK_REPO_SLUG}@${SNAPSHOT_TARGET_BRANCH}: $snapshot_commit"
    write_github_env GITHUB_SNAPSHOT_SYNC_STATUS pushed
    write_github_env GITHUB_SNAPSHOT_SYNC_COMMIT "$snapshot_commit"
    exit 0
  fi

  if [[ "$attempt" -lt "$SNAPSHOT_MAX_PUSH_ATTEMPTS" ]]; then
    echo "official baseline publication push failed; retrying with fresh ${BENCHMARK_REPO_REMOTE}/${SNAPSHOT_TARGET_BRANCH} in ${SNAPSHOT_PUSH_RETRY_SECONDS}s (attempt $attempt/$SNAPSHOT_MAX_PUSH_ATTEMPTS)" >&2
    sleep "$SNAPSHOT_PUSH_RETRY_SECONDS"
  fi
done

echo "failed to push official baseline publication after $SNAPSHOT_MAX_PUSH_ATTEMPTS attempts" >&2
exit 1