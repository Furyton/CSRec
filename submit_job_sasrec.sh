#!/bin/bash

SAVED_PATH=___slurm_log

JOB_NAME=test_el_sas_soft_2022
# JOB_NAME=lfms_base_sas_2020
# JOB_NAME=yelp_soft_sas
# JOB_NAME=yelp_dvae_sas
# JOB_NAME=yelp_ed_sas
# JOB_NAME=yelp_ad_sas
# JOB_NAME=lfms_sas_soft_test

CONFIG_PATH="config2.electronics/sasrec/config_test.json"
# CONFIG_PATH="config2.lastfm_small/sasrec/config.json"
# CONFIG_PATH="config2.yelp/sasrec/config_distill.json"
# CONFIG_PATH="config2.yelp/sasrec/config_dvae.json"
# CONFIG_PATH="config2.yelp/sasrec/config_distill_ensemble.json"
# CONFIG_PATH="config2.yelp/sasrec/config_dvae_ensemble.json"
# CONFIG_PATH="config2.lastfm_small/sasrec/config_test.json"

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

sbatch --output=./$SAVED_PATH/$JOB_NAM.%j.out --error=./$SAVED_PATH/$JOB_NAME.%j.err --job-name=$JOB_NAME --export=ALL,CONFIG_PATH=$CONFIG_PATH,JOB_NAME=$JOB_NAME,SAVED_PATH=$SAVED_PATH sbatch_normal_script.bash

# sbatch --output=./$SAVED_PATH/${JOB_NAME}_%A_%a.%j.out --error=./$SAVED_PATH/${JOB_NAME}_%A_%a.%j.err --job-name=$JOB_NAME --array=$ARRAY_CONFIG --export=ALL,CONFIG_PATH=$CONFIG_PATH,JOB_NAME=$JOB_NAME,SAVED_PATH=$SAVED_PATH,HPARAMS_FILE=$HPARAMS_FILE sbatch_script.bash
