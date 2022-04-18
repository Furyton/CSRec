#!/bin/bash

SAVED_PATH=___slurm_log3

HPARAMS_FILE=partial_base_hyperparameters.txt
ARRAY_CONFIG=1-4%1

JOB_NAME=el_gru_base_partial_2022
# JOB_NAME=lfms_base_gru
# JOB_NAME=lfms_soft_gru
# JOB_NAME=el_dvae_gru
# JOB_NAME=lfms_ed_gru
# JOB_NAME=video_ad_gru

#CONFIG_PATH="config2.movies/gru4rec/config_dvae.json"
# CONFIG_PATH="config2.lastfm_small/gru4rec/config_partial.json"
# CONFIG_PATH="config2.lastfm_small/gru4rec/config_distill.json"
CONFIG_PATH="config2.electronics/gru4rec/config_partial.json"
# CONFIG_PATH="config2.lastfm_small/gru4rec/config_distill_ensemble.json"
# CONFIG_PATH="config2.yelp/gru4rec/config_dvae_ensemble.json"


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

echo hparams file: $HPARAMS_FILE

# echo submit to $GPU

echo start submitting job...

# sbatch --output=./$SAVED_PATH/$JOB_NAME.%j.out --error=./$SAVED_PATH/$JOB_NAME.%j.err --job-name=$JOB_NAME --nodelist=$GPU --export=ALL,CONFIG_PATH=$CONFIG_PATH,JOB_NAME=$JOB_NAME,SAVED_PATH=$SAVED_PATH sbatch_script.bash

# sbatch --output=./$SAVED_PATH/$JOB_NAME.%j.out --error=./$SAVED_PATH/$JOB_NAME.%j.err --job-name=$JOB_NAME --export=ALL,CONFIG_PATH=$CONFIG_PATH,JOB_NAME=$JOB_NAME,SAVED_PATH=$SAVED_PATH sbatch_script.bash

sbatch --output=./$SAVED_PATH/${JOB_NAME}_%A_%a.%j.out --error=./$SAVED_PATH/${JOB_NAME}_%A_%a.%j.err --job-name=$JOB_NAME --array=$ARRAY_CONFIG --export=ALL,CONFIG_PATH=$CONFIG_PATH,JOB_NAME=$JOB_NAME,SAVED_PATH=$SAVED_PATH,HPARAMS_FILE=$HPARAMS_FILE sbatch_script.bash
