#!/bin/bash
# Unified data generation script for all PDEs.
#
# Due to dolfinx memory constraints, each process generates a limited number
# of samples. This script launches multiple processes with different --count
# values to accumulate the desired total dataset.
#
# Usage:
#   bash scripts/generate_data.sh <pde> <mode> <num_processes>
#   bash scripts/generate_data.sh all train 8       # all PDEs, 8 processes each
#   bash scripts/generate_data.sh laplace2d test 8  # single PDE, test mode
#
# Run from the sni/ directory.

set -e

PDE="${1:?Usage: $0 <pde|all> <train|test> <num_processes>}"
MODE="${2:?Usage: $0 <pde|all> <train|test> <num_processes>}"
NUM_PROCESSES="${3:-1}"

SCRIPT="data_generation/generate.py"

if [ "$PDE" = "all" ]; then
    PDES=("laplace2d" "laplace2d_mixed" "darcy2d" "heat2d" "nonlinear_poisson2d")
else
    PDES=("$PDE")
fi

for pde in "${PDES[@]}"; do
    echo "=== Generating ${pde} (${MODE}) with ${NUM_PROCESSES} process(es) ==="
    for ((n=1; n<=NUM_PROCESSES; n++)); do
        echo "  Process ${n}/${NUM_PROCESSES}..."
        python "$SCRIPT" --pde "$pde" --mode "$MODE" --count "$n"
    done
    echo "  Merging shards..."
    python "$SCRIPT" --pde "$pde" --mode "$MODE" --merge
    echo "  Done: ${pde}"
done

echo "=== All done ==="
