#!/bin/bash
#SBATCH --job-name=exp2
#SBATCH --time=14:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=1
#SBATCH --mem=20G
echo Hello
sleep 10
echo Starting

module load Julia/1.9.3-linux-x86_64
stdbuf -o0 -e0 /mnt/appl/software/Julia/1.9.3-linux-x86_64/bin/julia  --color=no -O3 artificial.jl --dataset $1 --task $2 --incarnation $3 #-i $SLURM_ARRAY_TASK_ID