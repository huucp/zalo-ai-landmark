#!/usr/bin/env bash

nohup python3 train.py --net_type resnet --depth 101 -fu 6 -rs 1 --num_epochs 70 --num_workers 4 > resnet_101_rs1.log 2>&1 &
nohup python3 train.py --net_type resnet --depth 101 -fu 6 -rs 2 --num_epochs 50 --num_workers 4 --flip .3 --brightness .4 --contrast .4 --saturation .4 --hue .1 > resnet_101_rs2.log 2>&1 &
nohup python3 train.py --net_type resnet --depth 152 -fu 6 -rs 3 --num_epochs 70 > resnet_152_rs3.log 2>&1 &
nohup python3 train.py --net_type xception -fu 17 -rs 5 --num_epochs 80 > xception_rs4.log 2>&1 &
nohup python3 train.py --net_type inception_v3 -fu 15 -rs 7 --num_epochs 80 > inceptionv3_rs5.log 2>&1 &
nohup python3 train.py --net_type resnet --skip 31,65 --depth 50 -fu 6 -rs 11 --num_epochs 60 --num_workers 6 > resnet_50_rs11.log 2>&1 &
nohup python3 train.py --net_type resnet --skip 1,58 --depth 50 -fu 6 -rs 13 --num_epochs 40 --num_workers 6 > resnet_50_rs13.log 2>&1 &
