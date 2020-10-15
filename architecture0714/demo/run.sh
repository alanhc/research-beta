#!/bin/bash

echo "start"
eval "$(conda shell.bash hook)"

echo "step 1"
conda activate research
python 1_run-make-input.py

echo "step 2"
conda activate tf2
python 2_run-make-day_img.py
echo "step 3"
conda activate research
python 3_yolo.py

echo "step 4"
conda activate research 
python 4_imageEnhance.py

echo "step 5"
python 5_make_result.py
