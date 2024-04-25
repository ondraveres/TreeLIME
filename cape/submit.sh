#!/bin/bash
#hepatitis deviceid mutagenesis
for counter in {1..3}; do
    if [ $counter -le 100 ]; then
        echo "test"
    fi
    sbatch -J task_${counter} -D ${PWD} slurm.sh $counter
    counter=$((counter+1))
    #exit 0
done
