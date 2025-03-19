#!/bin/bash
#SBATCH --job-name=mmmu_txt

#SBATCH --partition=a100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48GB
#SBATCH --time=12:00:00
#SBATCH --output=mmmu_long_txt.out


# first we ensure a clean running environment:
module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda init
conda activate /gpfs/scratch/vm2781/conda/envs/cambrian-base

python /gpfs/scratch/vm2781/cambrian/eval/eval/mmmu/shuffle/mmmu_eval_shuffle_txt.py   --model_path /gpfs/scratch/vm2781/cambrian/eval/cambrian-8b/ --answers_file  /gpfs/scratch/vm2781/cambrian/eval/answers/mmmu_txt.jsonl --conv_mode llama_3
#python /gpfs/home/vm2781/cambrian/eval/eval/mmmu/mmmu_test.py --answers_file  ./answers/answer_mmmu_0.jsonl --output_file ./answers/incorrect/incorrect_mmmu.jsonl --csv_file ./experiments_mmmu.csv --extra_outdir ./eval/eval/mmmu
#python ./scripts/tabulate.py --eval_dir answers --experiment_csv ./experiments.csv --out_pivot pivot.xlsx --out_all_results all_results.csv

echo "YOU CAN TELL ERRYBODY"
echo "Cache cleanup complete."

# python /gpfs/scratch/vm2781/cambrian/eval/eval/mmmu/mme_eval.py   --model_path /gpfs/scratch/vm2781/cambrian/eval/cambrian-8b/ --answers_file  /gpfs/scratch/vm2781/cambrian/eval/answers/mmmu_eval_print.jsonl --conv_mode llama_3
