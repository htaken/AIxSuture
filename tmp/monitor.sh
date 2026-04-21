#!/usr/bin/env bash
set -u
LOG="${1:-/home/takenouchi/AIxSuture/tmp/log/monitor.log}"
echo "timestamp,mem_total_mb,mem_used_mb,mem_free_mb,mem_available_mb,swap_used_mb,gpu_used_mib,gpu_free_mib,gpu_util_pct" > "$LOG"
while true; do
    TS=$(date +%Y-%m-%dT%H:%M:%S)
    MEM=$(awk '/^MemTotal:/{t=$2}/^MemAvailable:/{a=$2}/^MemFree:/{f=$2}END{printf "%d,%d,%d", t/1024, (t-a)/1024, f/1024}' /proc/meminfo)
    AVAIL=$(awk '/^MemAvailable:/{printf "%d", $2/1024}' /proc/meminfo)
    SWAP=$(awk '/^SwapTotal:/{t=$2}/^SwapFree:/{f=$2}END{printf "%d", (t-f)/1024}' /proc/meminfo)
    GPU=$(nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    echo "${TS},${MEM},${AVAIL},${SWAP},${GPU}" >> "$LOG"
    sleep 1
done
