#!/bin/bash
# scripts/watch_results.sh
# Polls results/tables/ for changes, regenerates metrics + figures.
set -euo pipefail

INTERVAL="${1:-60}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TABLES_DIR="$PROJECT_ROOT/results/tables"

CHECKSUM=$(mktemp)
trap "rm -f $CHECKSUM" EXIT
find "$TABLES_DIR" -name "*.csv" -exec md5sum {} \; 2>/dev/null | sort > "$CHECKSUM"

while true; do
    sleep "$INTERVAL"
    NEW=$(mktemp)
    find "$TABLES_DIR" -name "*.csv" -exec md5sum {} \; 2>/dev/null | sort > "$NEW"
    if ! diff -q "$CHECKSUM" "$NEW" > /dev/null 2>&1; then
        echo "$(date +%H:%M:%S) updating..."
        python "$PROJECT_ROOT/code/results_tracker.py" --phase all 2>&1 | tail -3
        python "$PROJECT_ROOT/code/visualize.py" --all 2>&1 | tail -3
        cp "$NEW" "$CHECKSUM"
    fi
    rm -f "$NEW"
done