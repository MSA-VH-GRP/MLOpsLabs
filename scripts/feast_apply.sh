#!/usr/bin/env bash
# Apply Feast feature definitions to the registry.
# Run from the project root after `pip install -e .`.

set -euo pipefail

cd "$(dirname "$0")/.."
echo "Running: feast apply (repo: feast/)"
feast -c feast/ apply
echo "Feast apply complete."
