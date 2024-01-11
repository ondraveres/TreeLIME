#!/bin/bash
#SBATCH --job-name=cuckoo
#SBATCH --time=24:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=1
#SBATCH --mem=20G

julia  --color=no -O3 explain_stats2.jl --name $1 --pruning_method $2  -i $3
