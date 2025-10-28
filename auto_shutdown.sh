#!/usr/bin/env bash
# Auto-shutdown after GPU inactivity
# Requires: nvidia-smi and runpodctl, plus RUNPOD_POD_ID set

# Usage:
# nohup sh auto_shutdown.sh & 
# To stop the script:
# pkill -f auto_shutdown.sh

# --- config ---
THRESHOLD=5        # % GPU utilization considered idle
IDLE_MINUTES=10    # how long to stay below threshold before stopping
CHECK_INTERVAL=60  # seconds between checks

# --- state ---
IDLE_SECS=0

echo "Monitoring GPU utilization... will stop pod after $IDLE_MINUTES minutes below ${THRESHOLD}%."

while true; do
  UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | awk '{sum+=$1} END {print sum/NR}')
  UTIL=${UTIL%.*}

  if [ "$UTIL" -le "$THRESHOLD" ]; then
    IDLE_SECS=$((IDLE_SECS + CHECK_INTERVAL))
    echo "$(date): GPU idle (${UTIL}%) for $((IDLE_SECS/60)) minutes..."
  else
    echo "$(date): GPU active (${UTIL}%), resetting idle timer."
    IDLE_SECS=0
  fi

  if [ "$IDLE_SECS" -ge $((IDLE_MINUTES*60)) ]; then
    echo "GPU idle for $IDLE_MINUTES minutes — stopping pod."
    runpodctl stop pod "$RUNPOD_POD_ID"
    exit 0
  fi

  sleep "$CHECK_INTERVAL"
done
