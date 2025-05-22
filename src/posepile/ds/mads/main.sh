#!/usr/bin/env bash
# Martial Arts, Dancing and Sports Dataset: a Challenging Stereo and Multi-View Dataset for 3D Human Pose Estimation.
# Weichen Zhang, Zhiguang Liu, Liuyang Zhou, Howard Leung, and Antoni B. Chan,
# Image and Vision Computing, 61:22-39, May 2017
# https://visal.cs.cityu.edu.hk/research/mads/

set -euo pipefail
source posepile/functions.sh
check_data_root

# Get data from https://drive.google.com/drive/folders/0B0AquUC4V8cFU2otR3l3WWRUVVk?resourcekey=0-KC-rxBAHiIIpylFRCTESNQ&usp=sharing

mkdircd "$DATA_ROOT/mads"

zip -s 0 MADS_multiview.zip --out single.zip
extractrm single.zip

python -m posepile.ds.mads.main --stage=1

# num_images=$(find "$DATA_ROOT/mads_downscaled" -name '*.jpg' | wc -l)
# should be around 32649
# Then we segment these downscaled images
for i in {0..3}; do
  SLURM_ARRAY_TASK_ID=$i python -m humcentr_cli.segment_people --image-root="$DATA_ROOT/mads_downscaled" --out-dir="$DATA_ROOT/mads_downscaled/masks"
done
python -m posepile.ds.mads.main --stage=2
