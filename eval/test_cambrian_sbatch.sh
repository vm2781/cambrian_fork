#!/bin/bash
#SBATCH --job-name=math_8b_img_ipc

#SBATCH --partition=a100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48GB
#SBATCH --time=06:00:00
#SBATCH --output=math_8b_img_ipc.out


# first we ensure a clean running environment:
module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda init
conda activate /gpfs/scratch/vm2781/conda/envs/cambrian-base

python /gpfs/scratch/vm2781/cambrian/eval/eval/mathvista/shuffle/mathvista_eval_img_ipc.py   --model_path /gpfs/scratch/vm2781/cambrian/eval/cambrian-8b/ --answers_file  /gpfs/scratch/vm2781/cambrian/eval/answers/ipc/mathvista/mathvista_8b_img.jsonl --conv_mode llama_3
echo "Cache cleanup complete."
