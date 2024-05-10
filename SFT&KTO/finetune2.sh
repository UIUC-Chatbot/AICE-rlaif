#!/bin/bash

#SBATCH --job-name="sft_large"            
#SBATCH --nodes=2                       ### Number of Nodes
#SBATCH --ntasks-per-node=1
#SBATCH --partition=a100                ### Cheaha Partition
#SBATCH --time=04:00:00                 ### Estimated Time of Completion, 4 hour
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=16
#SBATCH --output=sft_large.o%j          ### Slurm Output file
#SBATCH --error=sft_large.e%j           ### Slurm Error file

### set email notification
##SBATCH --mail-type=BEGIN, END, FAIL
##SBATCH --mail-user=qilong2@illinois.edu

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

source ../utils/miniconda3/etc/profile.d/conda.sh
conda activate rag

nodes=( $( scontrol show hostnames $SLURM_JOB_MODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

#srun torchrun \
#--nnodes 5 \
#--nproc_per_node 2 \
#--rdzv_id $RANDOM \
#--rdzv_backend c10d \
#--rdzv_endpoint $head_node_ip:29500 \
#./SFT.py

export ACCELERATE_USE_FSDP=1
export FSDP_CPU_RAM_EFFICIENT_LOADING=1
srun torchrun \
--nnodes 2 \
--nproc_per_node 2 \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint $head_node_ip:29500 \
./SFT.py \
--config fsdp_config.yaml

