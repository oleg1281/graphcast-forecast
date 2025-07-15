#!/bin/bash

cd /mnt/c/Users/obedenok/PycharmProjects/Graphcast

docker run --gpus all --rm -it \
  -v "$(pwd):/app" \
  -v "/mnt/z/NOAA/predict_noaa:/app/predictions" \
  -v "/mnt/z/NOAA/data_NOAA:/app/data_NOAA" \
  graphcast_main python3 main.py

