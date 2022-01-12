#!/bin/bash

SAVED_PATH=___slurm_log

JOB_NAME=lastfm_normal_caser
# JOB_NAME=debug_dvae_distill
# JOB_NAME=pop_gru_T_2_alpha_40_cont_1
# JOB_NAME=caser_soft_gru_T_4_alpha_25_cont_1

#CONFIG_PATH="config/config.json"
CONFIG_PATH="config/config_caser.json"
GPU=gpu08

if [[ ! -d "$SAVED_PATH" ]]
then
    mkdir "$SAVED_PATH"
fi

if [ $# -eq 1 ]
then
    GPU=$1
fi

echo job name: $JOB_NAME

echo config file path: $CONFIG_PATH

echo slurm log saved path: $SAVED_PATH

echo submit to $GPU

echo start submitting job...

sbatch --output=./$SAVED_PATH/$JOB_NAME.%j.out --error=./$SAVED_PATH/$JOB_NAME.%j.err --job-name=$JOB_NAME --nodelist=$GPU --export=ALL,CONFIG_PATH=$CONFIG_PATH,JOB_NAME=$JOB_NAME,SAVED_PATH=$SAVED_PATH sbatch_script.bash

# sbatch --output=./$SAVED_PATH/$JOB_NAME.%j.out --error=./$SAVED_PATH/$JOB_NAME.%j.err --job-name=$JOB_NAME --export=ALL,CONFIG_PATH=$CONFIG_PATH,JOB_NAME=$JOB_NAME,SAVED_PATH=$SAVED_PATH sbatch_script.bash
