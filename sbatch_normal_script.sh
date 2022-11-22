#!/bin/bash

#SBATCH --partition=si

#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00

python main.py --config_file=$CONFIG_FILE
