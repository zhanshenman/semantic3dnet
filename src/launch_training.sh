#!/usr/bin/env bash
mkdir ../dump
source ~/.bashrc
export CUDA_VISIBLE_DEVICES=0
nohup th train_point_cloud.lua &
