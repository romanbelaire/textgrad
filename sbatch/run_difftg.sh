#!/bin/bash
#################################################
## DiffTG run script (adapted from sample_sbatch.sh)
##
## Usage:
##   sbatch sbatch/run_difftg.sh <config.yaml> <output_dir>
##
## GSM8K full test: sbatch sbatch/run_gsm8k.sh
#################################################

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=100gb
#SBATCH --time=01-0:00:0
#SBATCH --constraint=nopreempt&a40|l40|a100
#SBATCH --mail-type=END
#SBATCH --output=%u.difftg.%j.out
#SBATCH --requeue

#SBATCH --partition=researchshort
#SBATCH --account=pradeepresearch
#SBATCH --qos=research-1-qos
#SBATCH --mail-user=rbelaire.2021@phdcs.smu.edu.sg
#SBATCH --job-name=difftg

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/difftg_env/bin/activate" ]]; then
  REPO_ROOT="${SCRIPT_DIR}"
elif [[ -f "${SCRIPT_DIR}/../difftg_env/bin/activate" ]]; then
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
  echo "run_difftg.sh: could not find difftg_env/bin/activate (SCRIPT_DIR=${SCRIPT_DIR})" >&2
  exit 1
fi
cd "${REPO_ROOT}"

CONFIG_FILE="${1:-configs/toy_inference.yaml}"
OUTPUT_DIR="${2:-outputs/toy_inference_v1}"

module purge
module load Python/3.9.21-GCCcore-13.3.0
module load CUDA/12.6.0 cuDNN/9.5.0.50-CUDA-12.6.0 OpenMPI/5.0.3-GCC-13.3.0

export NVCC_FLAGS="-allow-unsupported-compiler"

source difftg_env/bin/activate

mkdir -p "$OUTPUT_DIR"

nvidia-smi
nvcc --version

if [[ -f hf_api ]]; then
    export HF_HOME=/common/scratch/users/r/rbelaire.2021/
    export HF_TOKEN=$(cat hf_api)
    huggingface-cli login --token "$HF_TOKEN" || echo "CLI login failed, relying on HF_TOKEN env var"
fi

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:32"

LOG_FILE="${OUTPUT_DIR}/run.log"

echo "=========================================="
echo "DiffTG run"
echo "  Config:     ${CONFIG_FILE}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "=========================================="

# No inner `srun` here: the whole sbatch allocation is a single GPU task and
# wrapping with `srun --gres=gpu:1` inside the step causes "Unable to satisfy
# cpu bind request" on this cluster.
python -m src.difftg.main \
    "${CONFIG_FILE}" \
    "${OUTPUT_DIR}" 2>&1 | tee "${LOG_FILE}"

echo "Completed DiffTG run; results in ${OUTPUT_DIR}"
