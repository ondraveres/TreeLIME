#!/bin/bash
#hepatitis deviceid mutagenesis
for d in hepatitis deviceid; do
    for t in `ls ../../data/raw/${d}`; do 
        counter=0
        for c in `ls ../../data/raw/${d}/${t} | grep concept`; do 
            # if [ $counter -ge 5 ]; then
            #     #exit 0
            # fi
            i=${c##*/}
            i=${c%_concept.jsonl}
            sbatch -J ${d}_${t}_${i} -D ${PWD} slurm.sh $d $t $i
            counter=$((counter+1))
            #exit 0
        done
    done
done
