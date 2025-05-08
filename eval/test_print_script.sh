#!/bin/bash
#SBATCH --job-name=test_print

#SBATCH --partition=a100_dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=00:30:00
#SBATCH --output=test_print_mmvp.out


# first we ensure a clean running environment:
module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda init
conda activate /gpfs/scratch/vm2781/conda/envs/cambrian-base
# rm -rf ~/.cache/huggingface
# find /tmp -user vm2781 -delete

# Environment setup
module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda init
conda activate /gpfs/scratch/vm2781/conda/envs/cambrian-base

MODEL_DIR="/gpfs/scratch/vm2781/cambrian/eval/cambrian-8b/"
BENCHMARK="mmvp"
# ANSWERS_DIR="/gpfs/scratch/vm2781/cambrian/eval/answers/docvqa"

# Run evaluations
python /gpfs/scratch/vm2781/cambrian/eval/eval/${BENCHMARK}/${BENCHMARK}_eval.py \
  --model_path ${MODEL_DIR} \
  --answers_file /gpfs/scratch/vm2781/cambrian/eval/answers/${BENCHMARK}/test.jsonl \
  --text_shuffle False --image_shuffle False --conv_mode llama_3
echo "Cache cleanup complete."
