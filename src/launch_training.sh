#!/usr/bin/env bash
mkdir ../dump
source ~/.bashrc
rm nohup.out.old
mv nohup.out nohup.out.old
export CUDA_VISIBLE_DEVICES=0
nohup th train_point_cloud.lua &
