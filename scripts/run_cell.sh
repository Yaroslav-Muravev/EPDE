#!/usr/bin/env bash
# Run one ablation cell across all 14 thesis systems at 30 reps.
# Usage: run_cell.sh <cell>
# Where <cell> is one of:
#   legacy wape instab reg wape_instab wape_reg instab_reg new
set -euo pipefail

CELL="${1:?usage: run_cell.sh <cell>}"
SYSTEMS=(ac burgers_inviscid burgers_viscous kdv kdv_cossin ks
         lorenz lv ns ode pde_compound pde_divide vdp wave)

# Cap BLAS / OpenMP threads so 8 sibling containers don't
# oversubscribe the host. docker-compose sets these in the
# environment; the defaults here are a safety net.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"

cd /work

for SYS in "${SYSTEMS[@]}"; do
    echo "========== cell=${CELL} system=${SYS} =========="
    # --pipelines <cell> selects exactly this cell from
    # thesis_runner._PIPELINE_SETTINGS. Resume=True (default)
    # skips reps whose <cell>_rep<NN>.json already exists.
    python projects/thesis/run.py "${SYS}" \
        --reps 30 \
        --pipelines "${CELL}" \
        || echo "    cell=${CELL} system=${SYS} exited non-zero"
done

echo "========== cell=${CELL} sweep finished =========="
