#!/bin/bash

#SBATCH --partition=debug

#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --exclude=gpu04,gpu03

python main.py --config_file=$CONFIG_PATH
