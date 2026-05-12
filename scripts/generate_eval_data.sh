#!/bin/bash
# Evaluation data generation on pre-defined domains (A, B, C).
#
# Usage:
#   bash scripts/generate_eval_data.sh                          # all PDEs, all domains
#   bash scripts/generate_eval_data.sh laplace2d                # single PDE, all domains
#   bash scripts/generate_eval_data.sh laplace2d A              # single PDE, single domain
#   bash scripts/generate_eval_data.sh all all 200              # all PDEs, all domains, 200 samples
#
# Run from the sni/ directory.

set -e

PDE="${1:-all}"
DOMAIN="${2:-all}"
NUM_SAMPLES="${3:-100}"

SCRIPT="data_generation/generate_eval.py"

echo "=== Generating evaluation data: pde=${PDE}, domain=${DOMAIN}, samples=${NUM_SAMPLES} ==="
python "$SCRIPT" --pde "$PDE" --domain "$DOMAIN" --num_samples "$NUM_SAMPLES"
echo "=== All done ==="
