#!/bin/bash

GRACE_PERIOD=20

shutdown_pid() {
  PID_FILE=$1

  if [ ! -f "$PID_FILE" ]; then
    echo "PID file $PID_FILE not found — skipping"
    return
  fi

  PID=$(cat "$PID_FILE")

  # Verify process exists
  if ! ps -p $PID > /dev/null; then
    echo "PID $PID not running — removing stale PID file"
    rm "$PID_FILE"
    return
  fi

  # Verify it's actually vLLM
  CMD=$(ps -p $PID -o cmd=)
  if [[ "$CMD" != *"vllm serve"* ]]; then
    echo "PID $PID is NOT vLLM — refusing to kill"
    return
  fi

  echo "Sending SIGTERM to vLLM PID $PID"
  kill -TERM $PID

  for ((i=0; i<$GRACE_PERIOD; i++)); do
    if ! ps -p $PID > /dev/null; then
      echo "✓ PID $PID exited cleanly"
      rm "$PID_FILE"
      return
    fi
    sleep 1
  done

  echo "⚠ PID $PID did not exit — force killing"
  kill -KILL $PID
  rm "$PID_FILE"
}

shutdown_pid vllm_gpu0.pid
shutdown_pid vllm_gpu1.pid
