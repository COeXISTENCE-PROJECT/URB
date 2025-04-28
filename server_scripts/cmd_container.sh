#!/bin/bash

cd /app/
python3 -m venv /tmp/venv
source /tmp/venv/bin/activate
echo "--- Installing dependencies ---"
pip3 install --force-reinstall --no-cache-dir -r requirements.txt
echo "--- Running main.py ---"
python3 -u scripts/ippo_torchrl.py --id 1 --conf 1_ippo --net gargenville --seed 42
