#!/bin/bash
#PBS -q SQUID
#PBS --group=G15538
#PBS -l elapstim_req=120:00:00
#PBS -l gpunum_job=8
cd $PBS_O_WORKDIR
module load BaseGPU/2021
module load cudnn/8.2.0.53
source /system/apps/rhel8/cpu/Anaconda3/2020.11/etc/profile.d/conda.sh
conda activate test_RLF
nvidia-smi > nv.txt
python test.py > result0.txt     #プログラムの実行

