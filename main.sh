#!/bin/sh
#BSUB -gpu "num=1:mode=exclusive_process" 
#BSUB -n 1
#BSUB -q gpu2
#BSUB -o %J.out
#BSUB -e %J.err 
#BSUB -J run_gpu 
python main.py
