#!/bin/bash
set -euo pipefail
mkdir -p outputs/toy_inference_sweep
sbatch sbatch/run_difftg.sh configs/_sweep/K0_step0.005_n1.yaml outputs/toy_inference_sweep/K0_step0.005_n1
sbatch sbatch/run_difftg.sh configs/_sweep/K5_step0.005_n1.yaml outputs/toy_inference_sweep/K5_step0.005_n1
sbatch sbatch/run_difftg.sh configs/_sweep/K5_step0.005_n3.yaml outputs/toy_inference_sweep/K5_step0.005_n3
sbatch sbatch/run_difftg.sh configs/_sweep/K5_step0.01_n1.yaml outputs/toy_inference_sweep/K5_step0.01_n1
sbatch sbatch/run_difftg.sh configs/_sweep/K5_step0.01_n3.yaml outputs/toy_inference_sweep/K5_step0.01_n3
sbatch sbatch/run_difftg.sh configs/_sweep/K5_step0.05_n1.yaml outputs/toy_inference_sweep/K5_step0.05_n1
sbatch sbatch/run_difftg.sh configs/_sweep/K5_step0.05_n3.yaml outputs/toy_inference_sweep/K5_step0.05_n3
sbatch sbatch/run_difftg.sh configs/_sweep/K20_step0.005_n1.yaml outputs/toy_inference_sweep/K20_step0.005_n1
sbatch sbatch/run_difftg.sh configs/_sweep/K20_step0.005_n3.yaml outputs/toy_inference_sweep/K20_step0.005_n3
sbatch sbatch/run_difftg.sh configs/_sweep/K20_step0.01_n1.yaml outputs/toy_inference_sweep/K20_step0.01_n1
sbatch sbatch/run_difftg.sh configs/_sweep/K20_step0.01_n3.yaml outputs/toy_inference_sweep/K20_step0.01_n3
sbatch sbatch/run_difftg.sh configs/_sweep/K20_step0.05_n1.yaml outputs/toy_inference_sweep/K20_step0.05_n1
sbatch sbatch/run_difftg.sh configs/_sweep/K20_step0.05_n3.yaml outputs/toy_inference_sweep/K20_step0.05_n3
