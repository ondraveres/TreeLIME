#!/bin/bash
#SBATCH --job-name=exp2
#SBATCH --time=14:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=1
#SBATCH --mem=20G

julia  --color=no -O3 artificial.jl --dataset $1 --task $2 --incarnation $3 #-i $SLURM_ARRAY_TASK_ID


 BSON CSV Clustering CodecZlib DataFrames Distances Duff  Flux HierarchicalUtils HypothesisTests IterTools JSON JsonGrinder MLDataPattern Mill ParallelDataTransfer PrettyTables Serialization Setfield SparseArrays Statistics StatsBase Test  Zygote