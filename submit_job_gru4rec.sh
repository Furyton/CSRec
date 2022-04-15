#!/bin/bash

SAVED_PATH=___slurm_log3

#JOB_NAME=movies_gru_test_soft_2022
JOB_NAME=movies_gru_dvae_2020
#JOB_NAME=el_gru_dvae_2020
#JOB_NAME=test_el_gru_dvae_2022
# JOB_NAME=lfms_test_dvae_gru
#JOB_NAME=yelp_base_gru_partial_2022_2001
# JOB_NAME=lfms_base_gru_2020
# JOB_NAME=yelp_soft_gru
# JOB_NAME=yelp_dvae_gru
# JOB_NAME=yelp_ed_gru
# JOB_NAME=yelp_ad_gru
# JOB_NAME=lfms_gru_soft_test

#CONFIG_PATH="config2.movies/gru4rec/config_test.json"
CONFIG_PATH="config2.movies/gru4rec/config_dvae.json"
# CONFIG_PATH="config2.lastfm_small/gru4rec/config_partial.json"
# CONFIG_PATH="config2.lastfm_small/gru4rec/config.json"
# CONFIG_PATH="config2.yelp/gru4rec/config_dvae.json"
# CONFIG_PATH="config2.yelp/gru4rec/config_distill_ensemble.json"
# CONFIG_PATH="config2.yelp/gru4rec/config_dvae_ensemble.json"
# CONFIG_PATH="config2.lastfm_small/gru4rec/config_test.json"


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
