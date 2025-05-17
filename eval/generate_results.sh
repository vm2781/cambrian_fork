#!/bin/bash
#SBATCH --job-name=generate_results
#SBATCH --partition=a100_short
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=4:00:00
#SBATCH --output=generate_results.out

# Define lists
MODEL_SIZES=("8b" "13b" "34b")
BENCHMARKS=("ade" "chartqa" "gqa" "infovqa" "mmbench_cn" "mmbench_en" "mmstar" "mmvet" "mmvp")

# Environment setup
module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda init
conda activate /gpfs/scratch/vm2781/conda/envs/cambrian-base

# Iterate over combinations
for MODEL_SIZE in "${MODEL_SIZES[@]}"; do
  MODEL_DIR="/gpfs/scratch/vm2781/cambrian/eval/cambrian-${MODEL_SIZE}/"

  for BENCHMARK in "${BENCHMARKS[@]}"; do
    echo "Running evaluation for MODEL_SIZE=${MODEL_SIZE}, BENCHMARK=${BENCHMARK}"

    python /gpfs/scratch/vm2781/cambrian/eval/eval/${BENCHMARK}/${BENCHMARK}_test.py \
      --answers_file /gpfs/scratch/vm2781/cambrian/eval/answers/${BENCHMARK}/${MODEL_SIZE}_nrm_0.jsonl \
      --csv_file     /gpfs/scratch/vm2781/cambrian/eval/answers/${BENCHMARK}/${BENCHMARK}_${MODEL_SIZE}.csv
      # --output_file  /gpfs/scratch/vm2781/cambrian/eval/answers/incorrect/${BENCHMARK}_${MODEL_SIZE}_nrm.jsonl \
    echo "--finished nrm"

    python /gpfs/scratch/vm2781/cambrian/eval/eval/${BENCHMARK}/${BENCHMARK}_test.py \
      --answers_file /gpfs/scratch/vm2781/cambrian/eval/answers/${BENCHMARK}/${MODEL_SIZE}_txt_0.jsonl \
      --csv_file     /gpfs/scratch/vm2781/cambrian/eval/answers/${BENCHMARK}/${BENCHMARK}_${MODEL_SIZE}.csv
      # --output_file  /gpfs/scratch/vm2781/cambrian/eval/answers/incorrect/${BENCHMARK}_${MODEL_SIZE}_txt.jsonl \
    echo "--finished txt"

    python /gpfs/scratch/vm2781/cambrian/eval/eval/${BENCHMARK}/${BENCHMARK}_test.py \
      --answers_file /gpfs/scratch/vm2781/cambrian/eval/answers/${BENCHMARK}/${MODEL_SIZE}_img_0.jsonl \
      --csv_file     /gpfs/scratch/vm2781/cambrian/eval/answers/${BENCHMARK}/${BENCHMARK}_${MODEL_SIZE}.csv
      # --output_file  /gpfs/scratch/vm2781/cambrian/eval/answers/incorrect/${BENCHMARK}_${MODEL_SIZE}_img.jsonl \
    echo "--finished img"

    python /gpfs/scratch/vm2781/cambrian/eval/eval/${BENCHMARK}/${BENCHMARK}_test.py \
      --answers_file /gpfs/scratch/vm2781/cambrian/eval/answers/${BENCHMARK}/${MODEL_SIZE}_rdm_0.jsonl \
      --csv_file     /gpfs/scratch/vm2781/cambrian/eval/answers/${BENCHMARK}/${BENCHMARK}_${MODEL_SIZE}.csv
      # --output_file  /gpfs/scratch/vm2781/cambrian/eval/answers/incorrect/${BENCHMARK}_${MODEL_SIZE}_rdm.jsonl \
    echo "--finished rdm"

    echo "Completed evaluation for MODEL_SIZE=${MODEL_SIZE}, BENCHMARK=${BENCHMARK}"
  done
done

echo "All evaluations complete. Cache cleanup complete."

