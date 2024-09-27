#!/bin/bash

OUT=

for MODEL in \
    sweep4-daring_5M_ep=2 \
    sweep4-daring_5M_ep=4 \
    sweep4-daring_5M_ep=6 \
    sweep4-sudo6k_daring_5M_ep=2 \
    sweep4-sudo6k_daring_5M_ep=4 \
    sweep4-sudo6k_daring_5M_ep=6 \
    sweep4-sudo_daring_5M_ep=2 \
    sweep4-sudo_daring_5M_ep=4 \
    sweep4-sudo_daring_5M_ep=6 \
    sweep4-systemchatdedup_daring_5M_ep=2 \
    sweep4-systemchatdedup_daring_5M_ep=4 \
    sweep4-systemchatdedup_daring_5M_ep=6 \
    sweep4-openhermes_5M_ep=2 \
    sweep4-openhermes_5M_ep=4 \
    sweep4-openhermes_5M_ep=6 \
    sweep4-sudo6k_openhermes_5M_ep=2 \
    sweep4-sudo6k_openhermes_5M_ep=4 \
    sweep4-sudo6k_openhermes_5M_ep=6 \
    sweep4-sudo_openhermes_5M_ep=2 \
    sweep4-sudo_openhermes_5M_ep=4 \
    sweep4-sudo_openhermes_5M_ep=6 \
    sweep4-systemchatdedup_openhermes_5M_ep=2 \
    sweep4-systemchatdedup_openhermes_5M_ep=4 \
    sweep4-systemchatdedup_openhermes_5M_ep=6 \
    sweep4-systemchat_5M_ep=2 \
    sweep4-systemchat_5M_ep=4 \
    sweep4-systemchat_5M_ep=6 \
    sweep4-daring_10M_ep=1 \
    sweep4-daring_10M_ep=2 \
    sweep4-daring_10M_ep=3 \
    sweep4-sudo6k_daring_10M_ep=1 \
    sweep4-sudo6k_daring_10M_ep=2 \
    sweep4-sudo6k_daring_10M_ep=3 \
    sweep4-sudo_daring_10M_ep=1 \
    sweep4-sudo_daring_10M_ep=2 \
    sweep4-sudo_daring_10M_ep=3 \
    sweep4-systemchat_daring_10M_ep=1 \
    sweep4-systemchat_daring_10M_ep=2 \
    sweep4-systemchat_daring_10M_ep=3 \
    sweep4-systemchatdedup_daring_10M_ep=1 \
    sweep4-systemchatdedup_daring_10M_ep=2 \
    sweep4-systemchatdedup_daring_10M_ep=3 \
    sweep4-openhermes_10M_ep=1 \
    sweep4-openhermes_10M_ep=2 \
    sweep4-openhermes_10M_ep=3 \
    sweep4-sudo6k_openhermes_10M_ep=2 \
    sweep4-sudo6k_openhermes_10M_ep=3 \
    sweep4-sudo_openhermes_10M_ep=1 \
    sweep4-sudo_openhermes_10M_ep=2 \
    sweep4-sudo_openhermes_10M_ep=3 \
    sweep4-systemchatdedup_openhermes_10M_ep=1 \
    sweep4-systemchatdedup_openhermes_10M_ep=2 \
    sweep4-systemchatdedup_openhermes_10M_ep=3
do
    echo $MODEL
    ROW1=$(python show_result_model.py --answer_model $MODEL --judge_model gpt-4o-mini-2024-07-18 --benchmark arena-hard-v0.2 | tail -n 1)
    ROW2=$(python show_result_model.py --answer_model $MODEL --judge_model gpt-4o-mini-2024-07-18 --benchmark alpaca-arena | tail -n 1)
    OUT="$OUT\n$ROW1,$ROW2"
done

echo -e $OUT