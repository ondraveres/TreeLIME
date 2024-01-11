#!/bin/bash

for d in mutagenesis hepatitis  deviceid  ; do
	for t in `ls ../../data/raw/${d}` ; do 
		for c in `ls ../../data/raw/${d}/${t} | grep concept` ; do 
			i=${c##*/}
			i=${c%_concept.jsonl}
			sbatch -J ${d}_${t}_${i} -D ${PWD} slurm.sh $d $t $i
			#exit 0
		done
	done
done
