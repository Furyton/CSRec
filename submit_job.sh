#!/bin/bash

SAVED_PATH=__slurm_log_gru4rec

JOB_NAME=normal_gru_retrain_40_epoch
# JOB_NAME=pop_gru_T_2_alpha_40_cont_1
# JOB_NAME=caser_soft_gru_T_4_alpha_25_cont_1

CONFIG_PATH="config/pure_gru_config.json"
GPU=gpu03

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
