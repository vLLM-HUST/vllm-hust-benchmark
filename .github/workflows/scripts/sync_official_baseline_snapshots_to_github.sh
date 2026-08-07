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
ALLOW_EMPTY_SNAPSHOT_SOURCE=${ALLOW_EMPTY_SNAPSHOT_SOURCE:-0}
SNAPSHOT_MAX_PUSH_ATTEMPTS=${SNAPSHOT_MAX_PUSH_ATTEMPTS:-4}
SNAPSHOT_PUSH_RETRY_SECONDS=${SNAPSHOT_PUSH_RETRY_SECONDS:-5}
SNAPSHOT_MAX_FETCH_ATTEMPTS=${SNAPSHOT_MAX_FETCH_ATTEMPTS:-4}
SNAPSHOT_FETCH_RETRY_SECONDS=${SNAPSHOT_FETCH_RETRY_SECONDS:-5}
SNAPSHOT_COMMIT_MESSAGE=${SNAPSHOT_COMMIT_MESSAGE:-chore(data): publish official ascend baseline snapshots}
GIT_COMMITTER_NAME=${GIT_COMMITTER_NAME:-vLLM-HUST Benchmark Bot}
GIT_COMMITTER_EMAIL=${GIT_COMMITTER_EMAIL:-benchmark-bot@vllm-hust.local}
BENCHMARK_REPO_REMOTE=${BENCHMARK_REPO_REMOTE:-origin}
BENCHMARK_REPO_SLUG=${BENCHMARK_REPO_SLUG:-vLLM-HUST/vllm-hust-benchmark}
BENCHMARK_REPO_GH_TOKEN=${BENCHMARK_REPO_GH_TOKEN:-}
BENCHMARK_REPO_SSH_KEY=${BENCHMARK_REPO_SSH_KEY:-}
ARTIFACT_VALIDATOR=${ARTIFACT_VALIDATOR:-$SOURCE_BENCHMARK_REPO_DIR/scripts/validate-run-artifact.sh}

required_submission_files=(
  leaderboard_manifest.json
  run_leaderboard.json
  env-manifest.json
  pip-packages.json
  checksums.sha256
  STATUS
)
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

validate_retry_configuration() {
  case "$SNAPSHOT_MAX_FETCH_ATTEMPTS" in
    ''|*[!0-9]*|0*)
      echo "SNAPSHOT_MAX_FETCH_ATTEMPTS must be a positive integer" >&2
      return 2
      ;;
  esac
  case "$SNAPSHOT_FETCH_RETRY_SECONDS" in
    ''|*[!0-9]*)
      echo "SNAPSHOT_FETCH_RETRY_SECONDS must be a non-negative integer" >&2
      return 2
      ;;
  esac
}

fetch_target_branch_with_retry() {
  local phase=$1
  local attempt=1

  while (( attempt <= SNAPSHOT_MAX_FETCH_ATTEMPTS )); do
    if git -C "$TARGET_BENCHMARK_REPO_DIR" fetch \
      "$BENCHMARK_REPO_REMOTE" "$SNAPSHOT_TARGET_BRANCH"; then
      return 0
    fi
    if (( attempt == SNAPSHOT_MAX_FETCH_ATTEMPTS )); then
      echo "official baseline publication ${phase} fetch failed after ${SNAPSHOT_MAX_FETCH_ATTEMPTS} attempts" >&2
      return 1
    fi
    echo "official baseline publication ${phase} fetch failed; retrying in ${SNAPSHOT_FETCH_RETRY_SECONDS}s (attempt $attempt/$SNAPSHOT_MAX_FETCH_ATTEMPTS)" >&2
    sleep "$SNAPSHOT_FETCH_RETRY_SECONDS"
    attempt=$((attempt + 1))
  done
}

validate_retry_configuration

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

if [[ ! -f "$ARTIFACT_VALIDATOR" ]]; then
  echo "artifact validator not found: $ARTIFACT_VALIDATOR" >&2
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
  if [[ "$ALLOW_EMPTY_SNAPSHOT_SOURCE" == "1" ]]; then
    echo "No source submissions matched pattern '$SNAPSHOT_SOURCE_PATTERN'; skipping publication sync"
    write_github_env GITHUB_SNAPSHOT_SYNC_STATUS skipped-empty
    exit 0
  fi
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
  if [[ "$(tr -d '[:space:]' < "$source_submission_dir/STATUS")" != "OK" ]]; then
    echo "source submission STATUS is not OK: $source_submission_dir/STATUS" >&2
    exit 2
  fi
  if ! bash "$ARTIFACT_VALIDATOR" "$source_submission_dir"; then
    echo "source submission artifact validation failed: $source_submission_dir" >&2
    exit 2
  fi
done

relative_snapshot_dir="leaderboard-data/snapshots"
publication_staging_dir=$(mktemp -d "$TARGET_BENCHMARK_REPO_DIR/.official-publication.XXXXXX")
staged_submission_dir="$publication_staging_dir/submissions"
staged_snapshot_dir="$publication_staging_dir/snapshots"

# Invoked indirectly by the EXIT trap below.
# shellcheck disable=SC2317,SC2329
cleanup_publication_staging() {
  rm -rf "$publication_staging_dir"
}
trap cleanup_publication_staging EXIT

reset_publication_staging() {
  rm -rf "$publication_staging_dir" || return $?
  mkdir -p "$publication_staging_dir" || return $?
}

git -C "$TARGET_BENCHMARK_REPO_DIR" config user.name "$GIT_COMMITTER_NAME"
git -C "$TARGET_BENCHMARK_REPO_DIR" config user.email "$GIT_COMMITTER_EMAIL"
configure_push_remote

prepare_publication_commit() {
  local relative_submission_paths=()
  local source_submission_dir
  local run_id
  local target_submission_dir
  local file_name

  reset_publication_staging || return $?
  fetch_target_branch_with_retry prepare || return $?
  git -C "$TARGET_BENCHMARK_REPO_DIR" checkout -B "$SNAPSHOT_TARGET_BRANCH" "$BENCHMARK_REPO_REMOTE/$SNAPSHOT_TARGET_BRANCH" || return $?

  mkdir -p "$staged_submission_dir" || return $?
  if [[ -d "$TARGET_BENCHMARK_REPO_DIR/submissions" ]]; then
    cp -a "$TARGET_BENCHMARK_REPO_DIR/submissions/." "$staged_submission_dir/" || return $?
  fi
  for source_submission_dir in "${source_submission_dirs[@]}"; do
    run_id=$(basename "$source_submission_dir")
    target_submission_dir="$staged_submission_dir/$run_id"
    relative_submission_paths+=("submissions/$run_id")

    rm -rf "$target_submission_dir" || return $?
    mkdir -p "$target_submission_dir" || return $?
    for file_name in "${required_submission_files[@]}"; do
      cp "$source_submission_dir/$file_name" "$target_submission_dir/$file_name" || return $?
    done
  done

  VLLM_HUST_BENCHMARK_REPO="$TARGET_BENCHMARK_REPO_DIR" \
  VLLM_HUST_WEBSITE_REPO="$WEBSITE_REPO_DIR" \
  PYTHONPATH="$TARGET_BENCHMARK_REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" -m vllm_hust_benchmark.cli publish-website \
      --source-dir "$staged_submission_dir" \
      --output-dir "$staged_snapshot_dir" \
      --execute || return $?

  for file_name in "${required_snapshot_files[@]}"; do
    if [[ ! -f "$staged_snapshot_dir/$file_name" ]]; then
      echo "missing generated snapshot file: $staged_snapshot_dir/$file_name" >&2
      return 2
    fi
  done

  if ! PYTHONPATH="$TARGET_BENCHMARK_REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" "$TARGET_BENCHMARK_REPO_DIR/scripts/validate_public_leaderboard_snapshots.py" \
    --snapshot-dir "$staged_snapshot_dir"; then
    echo "official baseline publication failed at public snapshot validation" >&2
    return 2
  fi
  if ! PYTHONPATH="$TARGET_BENCHMARK_REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" - "$staged_snapshot_dir" <<'PY'
import sys
from pathlib import Path

from vllm_hust_benchmark.integration import validate_public_snapshot_trend_admission

validate_public_snapshot_trend_admission(Path(sys.argv[1]))
PY
  then
    echo "official baseline publication failed at trend validation" >&2
    return 2
  fi

  mkdir -p "$TARGET_BENCHMARK_REPO_DIR/submissions" "$SNAPSHOT_OUTPUT_DIR" || return $?
  for source_submission_dir in "${source_submission_dirs[@]}"; do
    run_id=$(basename "$source_submission_dir")
    target_submission_dir="$TARGET_BENCHMARK_REPO_DIR/submissions/$run_id"
    rm -rf "$target_submission_dir" || return $?
    mkdir -p "$target_submission_dir" || return $?
    for file_name in "${required_submission_files[@]}"; do
      cp "$source_submission_dir/$file_name" "$target_submission_dir/$file_name" || return $?
    done
  done
  for file_name in "${required_snapshot_files[@]}"; do
    cp "$staged_snapshot_dir/$file_name" "$SNAPSHOT_OUTPUT_DIR/$file_name" || return $?
  done

  if [[ -n "$LOCAL_SNAPSHOT_OUTPUT_DIR" ]]; then
    mkdir -p "$LOCAL_SNAPSHOT_OUTPUT_DIR"
    for file_name in "${required_snapshot_files[@]}"; do
      cp "$SNAPSHOT_OUTPUT_DIR/$file_name" "$LOCAL_SNAPSHOT_OUTPUT_DIR/$file_name" || return $?
    done
  fi

  git -C "$TARGET_BENCHMARK_REPO_DIR" add "${relative_submission_paths[@]}" "$relative_snapshot_dir" || return $?
  if git -C "$TARGET_BENCHMARK_REPO_DIR" diff --cached --quiet; then
    return 3
  else
    diff_status=$?
    if [[ "$diff_status" -ne 1 ]]; then
      return "$diff_status"
    fi
  fi

  git -C "$TARGET_BENCHMARK_REPO_DIR" commit -m "$SNAPSHOT_COMMIT_MESSAGE" || return $?
}

verify_published_state() {
  local expected_commit=$1
  local verified_commit
  local source_submission_dir
  local run_id
  local file_name
  local checksum_file
  local expected_digest
  local relative_path
  local remote_path
  local local_digest
  local remote_blob

  compare_remote_file() {
    local commit=$1
    local path=$2
    local expected_file=$3
    if ! git -C "$TARGET_BENCHMARK_REPO_DIR" cat-file -e "$commit:$path"; then
      echo "official baseline publication verification failed: missing $path" >&2
      return 1
    fi
    remote_blob=$(mktemp)
    if ! git -C "$TARGET_BENCHMARK_REPO_DIR" show "$commit:$path" >"$remote_blob"; then
      rm -f "$remote_blob"
      echo "official baseline publication verification failed: unable to read $path" >&2
      return 1
    fi
    if ! cmp -s "$expected_file" "$remote_blob"; then
      rm -f "$remote_blob"
      echo "official baseline publication verification failed: content mismatch for $path" >&2
      return 1
    fi
    rm -f "$remote_blob"
  }

  fetch_target_branch_with_retry verify || return $?
  verified_commit=$(git -C "$TARGET_BENCHMARK_REPO_DIR" rev-parse "$BENCHMARK_REPO_REMOTE/$SNAPSHOT_TARGET_BRANCH") || return $?
  if [[ "$verified_commit" != "$expected_commit" ]]; then
    echo "official baseline publication verification failed: expected $expected_commit, got $verified_commit" >&2
    return 1
  fi

  for source_submission_dir in "${source_submission_dirs[@]}"; do
    run_id=$(basename "$source_submission_dir")
    for file_name in "${required_submission_files[@]}"; do
      compare_remote_file \
        "$verified_commit" \
        "submissions/$run_id/$file_name" \
        "$source_submission_dir/$file_name" || return 1
    done
    checksum_file="$source_submission_dir/checksums.sha256"
    while read -r expected_digest relative_path _; do
      [[ -z "$expected_digest" && -z "$relative_path" ]] && continue
      relative_path=${relative_path#./}
      if [[ ! "$expected_digest" =~ ^[[:xdigit:]]{64}$ ]] || [[ -z "$relative_path" ]] ||
        [[ "$relative_path" = /* ]] || [[ "$relative_path" == *".."* ]]; then
        echo "official baseline publication verification failed: invalid checksum entry in $checksum_file" >&2
        return 1
      fi
      remote_path="submissions/$run_id/$relative_path"
      if [[ ! -f "$source_submission_dir/$relative_path" ]]; then
        echo "official baseline publication verification failed: checksum references missing local file $relative_path" >&2
        return 1
      fi
      local_digest=$(sha256sum "$source_submission_dir/$relative_path" | awk '{print $1}')
      if [[ "$local_digest" != "$expected_digest" ]]; then
        echo "official baseline publication verification failed: local checksum mismatch for $relative_path" >&2
        return 1
      fi
      remote_blob=$(mktemp)
      if ! git -C "$TARGET_BENCHMARK_REPO_DIR" cat-file -e "$verified_commit:$remote_path" ||
        ! git -C "$TARGET_BENCHMARK_REPO_DIR" show "$verified_commit:$remote_path" >"$remote_blob"; then
        rm -f "$remote_blob"
        echo "official baseline publication verification failed: missing $remote_path" >&2
        return 1
      fi
      local_digest=$(sha256sum "$remote_blob" | awk '{print $1}')
      rm -f "$remote_blob"
      if [[ "$local_digest" != "$expected_digest" ]]; then
        echo "official baseline publication verification failed: remote checksum mismatch for $remote_path" >&2
        return 1
      fi
    done < "$checksum_file"
  done
  for file_name in "${required_snapshot_files[@]}"; do
    compare_remote_file \
      "$verified_commit" \
      "$relative_snapshot_dir/$file_name" \
      "$staged_snapshot_dir/$file_name" || return 1
  done

  write_github_env GITHUB_SNAPSHOT_SYNC_VERIFICATION verified
  write_github_env GITHUB_SNAPSHOT_SYNC_VERIFIED_COMMIT "$verified_commit"
  echo "Verified official baseline publication at ${BENCHMARK_REPO_SLUG}@${SNAPSHOT_TARGET_BRANCH}: $verified_commit"
}

write_publication_identity() {
  local submission_paths=""
  local source_submission_dir
  local run_id

  for source_submission_dir in "${source_submission_dirs[@]}"; do
    run_id=$(basename "$source_submission_dir")
    if [[ -n "$submission_paths" ]]; then
      submission_paths+=","
    fi
    submission_paths+="submissions/$run_id"
  done

  write_github_env GITHUB_SNAPSHOT_SYNC_REPO "$BENCHMARK_REPO_SLUG"
  write_github_env GITHUB_SNAPSHOT_SYNC_BRANCH "$SNAPSHOT_TARGET_BRANCH"
  write_github_env GITHUB_SNAPSHOT_SYNC_SUBMISSION_PATHS "$submission_paths"
  write_github_env GITHUB_SNAPSHOT_SYNC_SNAPSHOT_PATH "$relative_snapshot_dir"
}

for attempt in $(seq 1 "$SNAPSHOT_MAX_PUSH_ATTEMPTS"); do
  if prepare_publication_commit; then
    snapshot_commit=$(git -C "$TARGET_BENCHMARK_REPO_DIR" rev-parse HEAD)
    if git -C "$TARGET_BENCHMARK_REPO_DIR" push "$BENCHMARK_REPO_REMOTE" "HEAD:$SNAPSHOT_TARGET_BRANCH"; then
      write_github_env GITHUB_SNAPSHOT_SYNC_STATUS pushed
      write_github_env GITHUB_SNAPSHOT_SYNC_COMMIT "$snapshot_commit"
      write_publication_identity
      if verify_published_state "$snapshot_commit"; then
        echo "Pushed official baseline publication to ${BENCHMARK_REPO_SLUG}@${SNAPSHOT_TARGET_BRANCH}: $snapshot_commit"
      else
        verification_status=$?
        write_github_env GITHUB_SNAPSHOT_SYNC_VERIFICATION failed
        echo "official baseline publication push succeeded, but verification failed for $snapshot_commit" >&2
        exit "$verification_status"
      fi
      exit 0
    fi

    if [[ "$attempt" -lt "$SNAPSHOT_MAX_PUSH_ATTEMPTS" ]]; then
      echo "official baseline publication push failed; retrying with fresh ${BENCHMARK_REPO_REMOTE}/${SNAPSHOT_TARGET_BRANCH} in ${SNAPSHOT_PUSH_RETRY_SECONDS}s (attempt $attempt/$SNAPSHOT_MAX_PUSH_ATTEMPTS)" >&2
      sleep "$SNAPSHOT_PUSH_RETRY_SECONDS"
      continue
    fi
    break
  else
    prepare_status=$?
    if [[ "$prepare_status" -eq 3 ]]; then
      remote_commit=$(git -C "$TARGET_BENCHMARK_REPO_DIR" rev-parse "$BENCHMARK_REPO_REMOTE/$SNAPSHOT_TARGET_BRANCH")
      echo "Official baseline publication is already up to date on ${BENCHMARK_REPO_SLUG}@${SNAPSHOT_TARGET_BRANCH}"
      write_github_env GITHUB_SNAPSHOT_SYNC_STATUS unchanged
      write_github_env GITHUB_SNAPSHOT_SYNC_COMMIT "$remote_commit"
      write_publication_identity
      verify_published_state "$remote_commit"
      exit 0
    fi
    write_github_env GITHUB_SNAPSHOT_SYNC_STATUS rejected
    exit "$prepare_status"
  fi
done

echo "failed to push official baseline publication after $SNAPSHOT_MAX_PUSH_ATTEMPTS attempts" >&2
exit 1
