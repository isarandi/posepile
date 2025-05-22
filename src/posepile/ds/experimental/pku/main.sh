#!/usr/bin/env bash
# @article{liu2017pku,
#  title={PKU-MMD: A Large Scale Benchmark for Continuous Multi-Modal Human Action Understanding},
#  author={Chunhui, Liu and Yueyu, Hu and Yanghao, Li and Sijie, Song and Jiaying, Liu},
#  journal={ACM Multimedia workshop},
#  year={2017}
#}
# https://www.icst.pku.edu.cn/struct/Projects/PKUMMD.html
# https://github.com/ECHO960/PKU-MMD
# https://arxiv.org/abs/1703.07475
set -euo pipefail
source posepile/functions.sh
check_data_root

mkdircd "$DATA_ROOT/pku"

# Download it
# TODO add commands for it

# Generate pose predictions
for i in {0..107}; do
  SLURM_ARRAY_TASK_ID=$i python -m humcentr_cli.estimate_3d_pose_video \
    --model-path=TODO \
    --video-dir="$DATA_ROOT/pku/RGB_VIDEO" \
    --output-dir="$DATA_ROOT/pku/pred" \
    --file-pattern='*.avi' \
    --videos-per-task=10 \
    --camera-intrinsics-file="$DATA_ROOT/pku/intrinsic_matrix.pkl"
done

#-------------------------
# STCN VOS segmentation to match people over time and across views
# Single person
./pku_list_videos.py

ssubmit -c4 --mem=10G --array=0-317 \
  python -m segment_ntu_or_pku \
  --video-dir="$DATA_ROOT/pku/RGB_VIDEO" --output-dir="$DATA_ROOT/pku/stcn_pred" \
  --file-list-file="$DATA_ROOT/ntu/single_person_videos.txt" \
  --mem-every=20 \
  --mem-size=20 \
  --max-persons=1

# Two people
ssubmit -c4 --mem=10G --array=0-317 \
  python -m segment_ntu_or_pku \
  --video-dir="$DATA_ROOT/pku/RGB_VIDEO" \
  --output-dir="$DATA_ROOT/pku/stcn_pred" \
  --file-list-file="$DATA_ROOT/ntu/two_person_videos.txt" \
  --mem-every=20 \
  --mem-size=20 \
  --max-persons=2

#-----------------

# Triangulate
ssubmit -c4 -o "$DATA_ROOT/pku/triangulation_logs/%a.out" --gres=gpu:0 --mem=30G --array=1-32 \
  python -m posepile.ds.experimental.pku.triangulate

# Recover global scale
python -m posepile.ds.experimental.pku.recover_scale

# Generate crops
ssubmit -c4 --gres=gpu:0 --mem=5G --array=0-133 \
  python -m posepile.ds.experimental.pku.main --stage=1

# num_images=$(find "$DATA_ROOT/pku_downscaled/" -name '*.jpg' | wc -l)
# should be around 933786
# Then we segment these downscaled images
# This could be done, but wasn't yet
ssubmit -c4 --mem=5G --array=0-93 \
  python -m humcentr_cli.segment_people \
  --image-root="$DATA_ROOT/pku_downscaled" \
  --out-dir="$DATA_ROOT/pku_downscaled/masks"

# Put all together
python -m posepile.ds.experimental.pku.main --stage=2
