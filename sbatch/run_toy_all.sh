#!/bin/bash
#################################################
## DiffTG — Toy inference sweep (all toy variants)
## Usage: sbatch sbatch/run_toy_all.sh [output_base_dir]
#################################################

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=100gb
#SBATCH --time=1-00:00:0
#SBATCH --constraint=nopreempt&a40|l40|a100
#SBATCH --mail-type=END
#SBATCH --output=%u.difftg.toy.%j.out
#SBATCH --requeue

#SBATCH --partition=researchshort
#SBATCH --account=pradeepresearch
#SBATCH --qos=research-1-qos
#SBATCH --mail-user=rbelaire.2021@phdcs.smu.edu.sg
#SBATCH --job-name=difftg-toy-all

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/difftg_env/bin/activate" ]]; then
  REPO_ROOT="${SCRIPT_DIR}"
elif [[ -f "${SCRIPT_DIR}/../difftg_env/bin/activate" ]]; then
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
  echo "run_toy_all.sh: could not find difftg_env/bin/activate (SCRIPT_DIR=${SCRIPT_DIR})" >&2
  exit 1
fi
cd "${REPO_ROOT}"

CONFIG_TEMPLATE="configs/toy_inference.yaml"
OUTPUT_BASE_DIR="${1:-outputs/toy_inference_all}"
TOY_VARIANTS=("digit_shift" "reverse_string" "sum_digits")

module purge
module load Python/3.9.21-GCCcore-13.3.0
module load CUDA/12.6.0 cuDNN/9.5.0.50-CUDA-12.6.0 OpenMPI/5.0.3-GCC-13.3.0

export NVCC_FLAGS="-allow-unsupported-compiler"

source difftg_env/bin/activate

mkdir -p "${OUTPUT_BASE_DIR}"

nvidia-smi
nvcc --version

if [[ -f hf_api ]]; then
    export HF_HOME=/common/scratch/users/r/rbelaire.2021/
    export HF_TOKEN=$(cat hf_api)
    huggingface-cli login --token "${HF_TOKEN}" || echo "CLI login failed, relying on HF_TOKEN env var"
fi

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:32"

echo "=========================================="
echo "DiffTG toy sweep (all variants)"
echo "  Config template: ${CONFIG_TEMPLATE}"
echo "  Output base dir: ${OUTPUT_BASE_DIR}"
echo "  Variants:        ${TOY_VARIANTS[*]}"
echo "=========================================="

for variant in "${TOY_VARIANTS[@]}"; do
    run_output_dir="${OUTPUT_BASE_DIR}/${variant}"
    run_log="${run_output_dir}/run.log"
    run_cfg="${run_output_dir}/toy_inference.${variant}.yaml"

    mkdir -p "${run_output_dir}"

    python - <<PY
from pathlib import Path

template = Path("${CONFIG_TEMPLATE}").read_text()
lines = template.splitlines()
found = False
for i, line in enumerate(lines):
    if line.strip().startswith("variant:"):
        lines[i] = "  variant: ${variant}"
        found = True
        break
if not found:
    raise RuntimeError("Could not find task.variant in config template.")
Path("${run_cfg}").write_text("\n".join(lines) + "\n")
PY

    echo "------------------------------------------"
    echo "Running variant: ${variant}"
    echo "  Config:     ${run_cfg}"
    echo "  Output dir: ${run_output_dir}"
    echo "------------------------------------------"

    python -m src.difftg.main \
        "${run_cfg}" \
        "${run_output_dir}" 2>&1 | tee "${run_log}"
done

echo "Completed DiffTG toy sweep; results in ${OUTPUT_BASE_DIR}"
