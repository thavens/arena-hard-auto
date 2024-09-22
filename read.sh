#!/bin/bash

OUT=

for MODEL in \
    sweep3-daring_2M_ep=1_unfreeze=false \
    sweep3-daring_2M_ep=1_unfreeze=true \
    sweep3-daring_2M_ep=5_unfreeze=false \
    sweep3-daring_2M_ep=5_unfreeze=true \
    sweep3-daring_2M_ep=10_unfreeze=false \
    sweep3-daring_2M_ep=10_unfreeze=true \
    sweep3-deita_2M_ep=1_unfreeze=false \
    sweep3-deita_2M_ep=1_unfreeze=true \
    sweep3-deita_2M_ep=5_unfreeze=false \
    sweep3-deita_2M_ep=5_unfreeze=true \
    sweep3-deita_2M_ep=10_unfreeze=false \
    sweep3-deita_2M_ep=10_unfreeze=true \
    sweep3-systemchat_2M_ep=1_unfreeze=false \
    sweep3-systemchat_2M_ep=1_unfreeze=true \
    sweep3-systemchat_2M_ep=5_unfreeze=false \
    sweep3-systemchat_2M_ep=5_unfreeze=true \
    sweep3-systemchat_2M_ep=10_unfreeze=false \
    sweep3-systemchat_2M_ep=10_unfreeze=true \
    sweep3-systemchatdedup_daring_2M_ep=1_unfreeze=false \
    sweep3-systemchatdedup_daring_2M_ep=1_unfreeze=true \
    sweep3-systemchatdedup_daring_2M_ep=5_unfreeze=false \
    sweep3-systemchatdedup_daring_2M_ep=5_unfreeze=true \
    sweep3-systemchatdedup_daring_2M_ep=10_unfreeze=false \
    sweep3-systemchatdedup_daring_2M_ep=10_unfreeze=true \
    sweep3-system_daring_2M_ep=1_unfreeze=false \
    sweep3-system_daring_2M_ep=1_unfreeze=true \
    sweep3-system_daring_2M_ep=5_unfreeze=false \
    sweep3-system_daring_2M_ep=5_unfreeze=true \
    sweep3-system_daring_2M_ep=10_unfreeze=false \
    sweep3-system_daring_2M_ep=10_unfreeze=true \
    sweep3-system_systemchatdedup_daring_2M_ep=1_unfreeze=false \
    sweep3-system_systemchatdedup_daring_2M_ep=1_unfreeze=true \
    sweep3-system_systemchatdedup_daring_2M_ep=5_unfreeze=false \
    sweep3-system_systemchatdedup_daring_2M_ep=5_unfreeze=true \
    sweep3-system_systemchatdedup_daring_2M_ep=10_unfreeze=false \
    sweep3-system_systemchatdedup_daring_2M_ep=10_unfreeze=true \
    sweep3-systemchatdedup_deita_2M_ep=1_unfreeze=false \
    sweep3-systemchatdedup_deita_2M_ep=1_unfreeze=true \
    sweep3-systemchatdedup_deita_2M_ep=5_unfreeze=false \
    sweep3-systemchatdedup_deita_2M_ep=5_unfreeze=true \
    sweep3-systemchatdedup_deita_2M_ep=10_unfreeze=false \
    sweep3-systemchatdedup_deita_2M_ep=10_unfreeze=true \
    sweep3-system_deita_2M_ep=1_unfreeze=false \
    sweep3-system_deita_2M_ep=1_unfreeze=true \
    sweep3-system_deita_2M_ep=5_unfreeze=false \
    sweep3-system_deita_2M_ep=5_unfreeze=true \
    sweep3-system_deita_2M_ep=10_unfreeze=false \
    sweep3-system_deita_2M_ep=10_unfreeze=true \
    sweep3-system_systemchatdedup_deita_2M_ep=1_unfreeze=false \
    sweep3-system_systemchatdedup_deita_2M_ep=1_unfreeze=true \
    sweep3-system_systemchatdedup_deita_2M_ep=5_unfreeze=false \
    sweep3-system_systemchatdedup_deita_2M_ep=5_unfreeze=true \
    sweep3-system_systemchatdedup_deita_2M_ep=10_unfreeze=false \
    sweep3-system_systemchatdedup_deita_2M_ep=10_unfreeze=true
do
    echo $MODEL
    OUT="$OUT\n$(python show_result_model.py --answer_model $MODEL --judge_model gpt-4o-mini-2024-07-18 | tail -n 1)"
done

echo -e $OUT