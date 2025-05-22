#!/usr/bin/env bash
#@InProceedings{Yu_2020_CVPR,
#author = {Yu, Zhixuan and Yoon, Jae Shin and Lee, In Kyu and Venkatesh, Prashanth and Park, Jaesik and Yu, Jihun and Park, Hyun Soo},
#title = {HUMBI: A Large Multiview Dataset of Human Body Expressions},
#booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
#year = {2020}
#}
# https://humbi-data.net/
# https://github.com/zhixuany/HUMBI
# https://arxiv.org/abs/1812.00281
set -euo pipefail
source posepile/functions.sh
check_data_root

mkdircd "$DATA_ROOT/humbi"

unzip subj*.zip
rm subj*.zip

# Create ignore list of broken images (600 are found to be broken)
# But apparently some are broken in minor ways that not all jpeg readers will complain about!
# So we need to be careful when processing
python -m posepile.tools.find_broken_jpegs "$DATA_ROOT/humbi" "$DATA_ROOT/humbi/ignore_images.txt"

# Alternatively, use a Slurm array job that unrolls to the following:
for i in {0..127}; do
  SLURM_ARRAY_TASK_ID=$i python -m humcentr_cli.detect_people \
    --image-root="$DATA_ROOT/humbi" \
    --ignore-paths-file="$DATA_ROOT/humbi/ignore_images.txt" \
    --out-path="$DATA_ROOT/humbi/yolov4_detections.pkl"
done

python -m posepile.tools.pickle_dict_merge "$DATA_ROOT"/humbi/yolov4_detections_*.pkl "$DATA_ROOT/humbi/yolov4_detections.pkl"
rm "$DATA_ROOT"/humbi/yolov4_detections_*.pkl

# Stage 1 picks which examples need to be rendered (adaptive, motion-based sampling) and created downscaled images
python -m posepile.ds.humbi.main --stage=1

# num_images=$(find "$DATA_ROOT/humbi_downscaled/" -name '*.jpg' | wc -l)
# should be around 1271076
# Then we segment these downscaled images
for i in {0..127}; do
  SLURM_ARRAY_TASK_ID=$i python -m humcentr_cli.segment_people --image-root="$DATA_ROOT/humbi_downscaled" --out-dir="$DATA_ROOT/humbi_downscaled/masks"
done

python -m posepile.ds.humbi.main --stage=2
