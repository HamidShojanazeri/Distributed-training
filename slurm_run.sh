#!/bin/bash

#SBATCH --job-name=T5-LM-train

#SBATCH --ntasks=2

#SBATCH --nodes=2

#SBATCH --gpus-per-task=2

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

srun python -m torch.distributed.run \
--nnodes 2 \
--nproc_per_node 2 \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint $head_node_ip:29500 \
../FSDP_BERT_torchrun.py