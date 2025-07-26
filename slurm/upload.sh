#!/bin/sh
# Define variables
DST_USER=bareeva
DST_HOST=vca-gpu-headnode
SRC_PATH=/data2/bareeva/Projects/grad-slingshot/models
DST_PATH=/home/fe/bareeva/projects/model_weights
MODEL=resnet_18_for_tiny_3_by_3.pth

# Upload the weights using scp
scp "${SRC_PATH}/${MODEL}" "${DST_USER}@${DST_HOST}:${DST_PATH}/"