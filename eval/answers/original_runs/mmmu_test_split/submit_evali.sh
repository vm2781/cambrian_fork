#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=5:00:00
#SBATCH --mem=4GB
#SBATCH --job-name=eval_ai
#SBATCH --output=eval_ai.out



# first we ensure a clean running environment:
python /gpfs/scratch/vm2781/cambrian/eval/answers/mmmu_test_split/evalai_generation.py