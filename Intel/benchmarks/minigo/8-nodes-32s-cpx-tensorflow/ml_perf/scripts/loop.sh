#!/bin/bash
while true
do
  time $*
  if [ -f "$ABORT_FILE" ]; then
    exit 0
  fi
  sleep 1
done
