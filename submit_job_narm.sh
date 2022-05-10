#!/bin/bash

SAVED_PATH=___slurm_log2

#JOB_NAME=lfms_narm_ed_2020
# JOB_NAME=yelp_narm_dvae_2020
# JOB_NAME=movies_narm_dvae_2020
JOB_NAME=el_narm_dvae_2020

#CONFIG_PATH="config2.lastfm_small/narm/config_distill_ensemble.json"
# CONFIG_PATH="config2.yelp/narm/config_dvae.json"
# CONFIG_PATH="config2.movies/narm/config_dvae.json"
CONFIG_PATH="config2.electronics/narm/config_dvae.json"


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

sbatch --output=./$SAVED_PATH/$JOB_NAME.%j.out --error=./$SAVED_PATH/$JOB_NAME.%j.err --job-name=$JOB_NAME --export=ALL,CONFIG_PATH=$CONFIG_PATH,JOB_NAME=$JOB_NAME,SAVED_PATH=$SAVED_PATH sbatch_normal_script.bash

# sbatch --output=./$SAVED_PATH/${JOB_NAME}_%A_%a.%j.out --error=./$SAVED_PATH/${JOB_NAME}_%A_%a.%j.err --job-name=$JOB_NAME --array=$ARRAY_CONFIG --export=ALL,CONFIG_PATH=$CONFIG_PATH,JOB_NAME=$JOB_NAME,SAVED_PATH=$SAVED_PATH,HPARAMS_FILE=$HPARAMS_FILE sbatch_script.bash
