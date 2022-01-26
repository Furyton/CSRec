#!/bin/bash

SAVED_PATH=___slurm_log

# JOB_NAME=lastfm_small_gru_dvae_pop_T_9_alpha_75_dvae_alpha_75
# JOB_NAME=yelp_nextitnet_soft_pop_T_3_alpha_75
# JOB_NAME=gowalla_base_caser
# JOB_NAME=gowalla_gru4rec_dvae_pop_gru4rec_T_3_alpha_75_dvae_alpha_75
# JOB_NAME=debug_dvae_distill
# JOB_NAME=pop_gru_T_2_alpha_40_cont_1
# JOB_NAME=caser_soft_gru_T_4_alpha_25_cont_1
# JOB_NAME=gowalla_nextitnet_soft_pop_T_3_alpha_75
# JOB_NAME=yelp_ed_pop_g_g_9_1_al_75
# JOB_NAME=yelp_d_pop_g_g_T_3_al_90_dal_75

# JOB_NAME=gowalla_ed_p_n_n_5_5_al_75_T_3

JOB_NAME=yelp_ed_p_g_g_5_5_al_75_T_3
# CONFIG_PATH="config.gowalla/nextitnet/config_distill_ensemble.json"
CONFIG_PATH="config.yelp/gru4rec/config_distill_ensemble.json"
# CONFIG_PATH="config.lastfm_small/gru4rec/config_dvae.json"
# CONFIG_PATH="config.gowalla/nextitnet/config_distill.json"
# CONFIG_PATH="config.gowalla/gru4rec/config_dvae.json"
# CONFIG_PATH="config.yelp/config_dvae.json"
# CONFIG_PATH="config.yelp/config_dvae_caser.json"
# CONFIG_PATH="config.yelp/config_dvae.json"
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

# sbatch --output=./$SAVED_PATH/$JOB_NAME.%j.out --error=./$SAVED_PATH/$JOB_NAME.%j.err --job-name=$JOB_NAME --nodelist=$GPU --export=ALL,CONFIG_PATH=$CONFIG_PATH,JOB_NAME=$JOB_NAME,SAVED_PATH=$SAVED_PATH sbatch_script.bash

sbatch --output=./$SAVED_PATH/$JOB_NAME.%j.out --error=./$SAVED_PATH/$JOB_NAME.%j.err --job-name=$JOB_NAME --export=ALL,CONFIG_PATH=$CONFIG_PATH,JOB_NAME=$JOB_NAME,SAVED_PATH=$SAVED_PATH sbatch_script.bash
