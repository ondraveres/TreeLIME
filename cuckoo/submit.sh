#!/bin/bash
for ((i=1;i<=179;i+=1)) ; do
	sbatch -J banz_$i -D ${PWD} banz LbyL_HArr $i
	sbatch -J gnn_$i -D ${PWD} gnn LbyL_HAdd $i
done
