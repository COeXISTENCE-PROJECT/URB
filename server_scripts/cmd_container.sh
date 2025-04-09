#!/bin/bash

cd /app/
echo "--- Installing dependencies ---"
pip3 install --force-reinstall --no-cache-dir -r requirements.txt
echo "--- Running main.py ---"
python -u scripts/ippo_torchrl.py --id 1 --conf 1_ippo --net gargenville --seed 42
