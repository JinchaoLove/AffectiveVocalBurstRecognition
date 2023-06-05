#!/bin/bash
# Run new experiment after finishing last `run_exp.sh` process.
PID=$(pgrep -f train.py)
while ps -p $PID &>/dev/null; do sleep 20; done;
python cv.py
