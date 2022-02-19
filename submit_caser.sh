#!/bin/bash

SAVED_PATH=___slurm_log

HPARAMS_FILE=dvae_hyperparameters_2.txt
ARRAY_CONFIG=1-6%3
# JOB_NAME=yelp_base_caser
# JOB_NAME=yelp_soft_caser
JOB_NAME=yelp_dvae_caser
# JOB_NAME=yelp_ed_caser
# JOB_NAME=yelp_ad_caser

# CONFIG_PATH="config2.yelp/caser/config.json"
# CONFIG_PATH="config2.yelp/caser/config_distill.json"
CONFIG_PATH="config2.yelp/caser/config_dvae.json"
# CONFIG_PATH="config2.yelp/caser/config_distill_ensemble.json"
# CONFIG_PATH="config2.yelp/caser/config_dvae_ensemble.json"


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

# sbatch --output=./$SAVED_PATH/$JOB_NAME_%A_%a.%j.out --error=./$SAVED_PATH/$JOB_NAME_%A_%a.%j.err --job-name=$JOB_NAME --export=ALL,CONFIG_PATH=$CONFIG_PATH,JOB_NAME=$JOB_NAME,SAVED_PATH=$SAVED_PATH sbatch_script.bash

sbatch --output=./$SAVED_PATH/${JOB_NAME}_%A_%a.%j.out --error=./$SAVED_PATH/${JOB_NAME}_%A_%a.%j.err --job-name=$JOB_NAME --array=$ARRAY_CONFIG --export=ALL,CONFIG_PATH=$CONFIG_PATH,JOB_NAME=$JOB_NAME,SAVED_PATH=$SAVED_PATH,HPARAMS_FILE=$HPARAMS_FILE sbatch_script.bash
