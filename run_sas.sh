#!/bin/bash

#SBATCH -e test.err

#SBATCH -o test.out
#SBATCH -J test_soft_rec

#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu07
#SBATCH --cpus-per-task=2

#SBATCH --time=2:00:00

#SBATCH --mem=10G

#source ${HOME}/.bashrc
#cap

bash run.sh


