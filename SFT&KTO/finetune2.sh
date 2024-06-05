#!/bin/bash

#SBATCH --job-name="sft_large"            
#SBATCH --nodes=2                       ### Number of Nodes
#SBATCH --ntasks-per-node=1
#SBATCH --partition=a100                ### Cheaha Partition
#SBATCH --time=4:00:00                 ### Estimated Time of Completion, 18 hour
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
huggingface-cli login --token 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

nodes=( $( scontrol show hostnames $SLURM_JOB_MODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

## For SFT
#export ACCELERATE_USE_FSDP=1
#export FSDP_CPU_RAM_EFFICIENT_LOADING=1
#srun torchrun \
#--nnodes 4 \
#--nproc_per_node 2 \
#--rdzv_id $RANDOM \
#--rdzv_backend c10d \
#--rdzv_endpoint $head_node_ip:29500 \
#./SFT2.py \
#--config fsdp_config2.yaml

## For KTO
export ACCELERATE_USE_FSDP=1
export FSDP_CPU_RAM_EFFICIENT_LOADING=1
srun torchrun \
--nnodes 2 \
--nproc_per_node 2 \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint $head_node_ip:29500 \
./KTO2.py \
--output_dir "./llama-3-70b-uiuc" \
--max_prompt_length 3072 \
--max_completion_length 3072 \
--learning_rate 0.0002 \
--lr_scheduler_type "constant" \
--num_train_epochs 1 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 2 \
--optim "adamw_torch" \
--logging_steps 10 \
--save_strategy "epoch" \
--max_grad_norm 0.3 \
--warmup_ratio 0.03 \
--beta 0.1 \
--desirable_weight 1.0 \
--undesirable_weight 1.0 \
--bf16 \
--tf32 true \
--gradient_checkpointing \
--torch_dtype "bfloat16" \
--fsdp "full_shard auto_wrap offload" \
--fsdp_config fsdp_config.json
