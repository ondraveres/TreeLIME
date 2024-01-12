#!/bin/bash
#SBATCH --job-name=exp2
#SBATCH --time=14:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=1
#SBATCH --mem=20G

stdbuf -o0 -e0 julia  --color=no -O3 artificial.jl --dataset $1 --task $2 --incarnation $3 #-i $SLURM_ARRAY_TASK_ID