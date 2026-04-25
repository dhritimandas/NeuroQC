#!/bin/bash
set -euo pipefail
python code/results_tracker.py --phase all
python code/visualize.py --all
echo "Updated results/ and figures/"