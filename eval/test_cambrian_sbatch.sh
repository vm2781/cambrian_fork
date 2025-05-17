#!/bin/bash
#SBATCH --job-name=34b_mmvp
#SBATCH --partition=a100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=7:30:00
#SBATCH --output=34b_mmvp.out

# Define model size variable (e.g., 8b, 13b, 34b)
MODEL_SIZE="34b"
BENCHMARK="mmvp"
# Set conversation mode based on MODEL_SIZE
case "$MODEL_SIZE" in
  "8b")
    CONV_MODE="llama_3"
    ;;
  "13b")
    CONV_MODE="vicuna_v1"
    ;;
  "34b")
    CONV_MODE="chatml_direct"
    ;;
  *)
    echo "Unsupported MODEL_SIZE: $MODEL_SIZE"
    exit 1
    ;;
esac

# Environment setup
module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda init
conda activate /gpfs/scratch/vm2781/conda/envs/cambrian-base

MODEL_DIR="/gpfs/scratch/vm2781/cambrian/eval/cambrian-${MODEL_SIZE}/"
# ANSWERS_DIR="/gpfs/scratch/vm2781/cambrian/eval/answers/docvqa"

# Run evaluations
python /gpfs/scratch/vm2781/cambrian/eval/eval/${BENCHMARK}/${BENCHMARK}_eval.py \
  --model_path ${MODEL_DIR} \
  --answers_file /gpfs/scratch/vm2781/cambrian/eval/answers/${BENCHMARK}/${MODEL_SIZE}_nrm.jsonl \
  --conv_mode ${CONV_MODE}
echo "FINISHED WITH NORM"

python /gpfs/scratch/vm2781/cambrian/eval/eval/${BENCHMARK}/${BENCHMARK}_eval.py \
  --model_path ${MODEL_DIR} \
  --answers_file /gpfs/scratch/vm2781/cambrian/eval/answers/${BENCHMARK}/${MODEL_SIZE}_txt.jsonl \
  --image_shuffle  --conv_mode ${CONV_MODE}
echo "FINISHED WITH TXT"

python /gpfs/scratch/vm2781/cambrian/eval/eval/${BENCHMARK}/${BENCHMARK}_eval.py \
  --model_path ${MODEL_DIR} \
  --answers_file /gpfs/scratch/vm2781/cambrian/eval/answers/${BENCHMARK}/${MODEL_SIZE}_img.jsonl \
  --text_shuffle  --conv_mode ${CONV_MODE}
echo "FINISHED WITH IMG"

python /gpfs/scratch/vm2781/cambrian/eval/eval/${BENCHMARK}/${BENCHMARK}_eval.py \
  --model_path ${MODEL_DIR} \
  --answers_file /gpfs/scratch/vm2781/cambrian/eval/answers/${BENCHMARK}/${MODEL_SIZE}_rdm.jsonl \
  --text_shuffle  --image_shuffle  --conv_mode ${CONV_MODE}
echo "FINISHED WITH RDM"

echo "Cache cleanup complete."
