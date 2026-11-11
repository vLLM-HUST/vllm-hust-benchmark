#!/usr/bin/env bash
# Run by a host administrator. Isolated, finite CANN kernel call timing capture.
set -euo pipefail
umask 077
mode=${1:---check}
tracefs=/sys/kernel/tracing
if [[ ! -f "$tracefs/kprobe_events" ]]; then
  echo 'A mounted host tracefs with kprobe_events is required.' >&2
  exit 2
fi
if [[ $(uname -m) != aarch64 ]]; then
  echo 'This capture uses AArch64 argument registers.' >&2
  exit 2
fi
symbols=(trs_hw_sq_send_task hal_kernel_trs_chan_send trs_chan_fill_sqe trs_chan_flush_sqe_cache)
for symbol in "${symbols[@]}"; do
  if ! awk -v symbol="$symbol" '$3 == symbol { found=1 } END { exit !found }' /proc/kallsyms; then
    echo "Kernel symbol unavailable: $symbol" >&2
    exit 3
  fi
done
printf 'Host: %s\nKernel: %s\n' "$(hostname)" "$(uname -r)"
printf 'Available target symbol: %s\n' "${symbols[@]}"
if [[ "$mode" == --check ]]; then
  echo 'Read-only precheck completed; no probes created.'
  exit 0
fi
if [[ "$mode" != --capture || $# -lt 2 ]]; then
  echo "Usage: $0 --check | --capture OUTPUT_DIRECTORY [SECONDS, max 30]" >&2
  exit 2
fi
seconds=${3:-20}
if [[ ! "$seconds" =~ ^[0-9]+$ ]] || (( seconds < 1 || seconds > 30 )); then
  echo 'Capture length must be 1..30 seconds.' >&2
  exit 2
fi
out=$2
if [[ -e "$out" ]]; then
  echo 'Output directory already exists; choose a new one.' >&2
  exit 2
fi
mkdir -p -- "$out"
out=$(realpath -- "$out")
group="pgrestore_${$}_$(date +%s)"
instance="$tracefs/instances/$group"
created=()
reader=''
cleanup() {
  set +e
  if [[ -f "$instance/tracing_on" ]]; then printf '0\n' > "$instance/tracing_on"; fi
  if [[ -n "$reader" ]]; then kill "$reader" 2>/dev/null; wait "$reader" 2>/dev/null; fi
  if [[ -f "$instance/events/$group/enable" ]]; then printf '0\n' > "$instance/events/$group/enable"; fi
  for event in "${created[@]}"; do printf -- '-:%s/%s\n' "$group" "$event" >> "$tracefs/kprobe_events"; done
  if [[ -d "$instance" ]]; then rmdir -- "$instance"; fi
}
trap cleanup EXIT INT TERM
mkdir -- "$instance"
printf '0\n' > "$instance/tracing_on"
printf 'mono\n' > "$instance/trace_clock"
printf '512\n' > "$instance/buffer_size_kb"
for index in "${!symbols[@]}"; do
  symbol=${symbols[$index]}
  entry="${group}_${index}_enter"
  leave="${group}_${index}_leave"
  extra=''
  # Exported prototype: (struct trs_id_inst *inst, int chan_id, ...).
  # Validate this register against the experiment-owned /proc/trs_chan ID.
  if (( index == 1 )); then extra=' chan_id=%x1:u32'; fi
  printf 'p:%s/%s %s arg0=%%x0:x64%s\n' "$group" "$entry" "$symbol" "$extra" >> "$tracefs/kprobe_events"
  created+=("$entry")
  printf 'r512:%s/%s %s retval=$retval:s64\n' "$group" "$leave" "$symbol" >> "$tracefs/kprobe_events"
  created+=("$leave")
done
{
  printf 'host=%s\nkernel=%s\nboot_id=%s\ngroup=%s\nseconds=%s\n' \
    "$(hostname)" "$(uname -r)" "$(cat /proc/sys/kernel/random/boot_id)" "$group" "$seconds"
  for index in "${!symbols[@]}"; do printf 'probe_%s=%s\n' "$index" "${symbols[$index]}"; done
  cat "$instance/trace_clock"
} > "$out/environment.txt"
cp -- "$0" "$out/executed-capture.sh"
mkdir -- "$out/event-formats" "$out/per-cpu-before" "$out/per-cpu-after"
cat "$instance/trace_options" > "$out/trace-options.txt"
for event in "${created[@]}"; do
  cat "$instance/events/$group/$event/format" > "$out/event-formats/$event.txt"
done
for stat in "$instance"/per_cpu/cpu*/stats; do
  cpu=$(basename -- "$(dirname -- "$stat")")
  cat "$stat" > "$out/per-cpu-before/$cpu.txt"
done
printf '1\n' > "$instance/events/$group/enable"
cat "$instance/trace_pipe" > "$out/kernel-calls.txt" &
reader=$!
printf '1\n' > "$instance/tracing_on"
printf 'Capture ready: %s; run the slow restore now.\n' "$group"
sleep "$seconds"
printf '0\n' > "$instance/tracing_on"
printf '0\n' > "$instance/events/$group/enable"
sleep .2
kill "$reader" 2>/dev/null || true
wait "$reader" 2>/dev/null || true
reader=''
for stat in "$instance"/per_cpu/cpu*/stats; do
  cpu=$(basename -- "$(dirname -- "$stat")")
  cat "$stat" > "$out/per-cpu-after/$cpu.txt"
done
awk -v group="$group" 'index($0, group)' "$tracefs/kprobe_profile" > "$out/probe-profile.txt"
wc -c -- "$out/kernel-calls.txt"
printf 'Capture completed: %s\n' "$out"
