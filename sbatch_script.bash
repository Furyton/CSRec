#!/bin/bash

#SBATCH --partition=debug

#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=30:00:00
#SBATCH --exclude=gpu03,gpu04

python main.py --task_id=$SLURM_ARRAY_TASK_ID --config_file=$CONFIG_PATH $(head -$SLURM_ARRAY_TASK_ID $HPARAMS_FILE | tail -1)

