#!/bin/bash
#SBATCH --job-name=mmmu_img_lmc

#SBATCH --partition=a100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48GB
#SBATCH --time=06:00:00
#SBATCH --output=mmmu_img_lmc.out


# first we ensure a clean running environment:
module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda init
conda activate /gpfs/scratch/vm2781/conda/envs/cambrian-base

python /gpfs/scratch/vm2781/cambrian/eval/eval/mmmu/shuffle/mmmu_eval_shuffle_img.py   --model_path /gpfs/scratch/vm2781/cambrian/eval/cambrian-13b/ --answers_file  /gpfs/scratch/vm2781/cambrian/eval/answers/language_model_change/mmmu_img.jsonl

echo "YOU CAN TELL ERRYBODY"
echo "Cache cleanup complete."
