#!/usr/bin/env bash
# @inproceedings{shahroudy2016ntu,
#  title={NTU RGB+D: A large scale dataset for 3D human activity analysis},
#  author={Shahroudy, Amir and Liu, Jun and Ng, Tian-Tsong and Wang, Gang},
#  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
#  year={2016}
#}
#
#@article{liu2020ntu,
#  title={NTU RGB+D 120: A large-scale benchmark for 3D human activity understanding},
#  author={Liu, Jun and Shahroudy, Amir and Perez, Mauricio and Wang, Gang and Duan, Ling-Yu and Kot, Alex C},
#  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
#  volume={42},
#  number={10},
#  pages={2684--2701},
#  year={2020}
#}
# https://github.com/shahroudy/NTURGB-D
set -euo pipefail
source posepile/functions.sh
check_data_root

mkdircd "$DATA_ROOT/ntu"

# Download it
# TODO commands for it

python -m posepile.ds.experimental.ntu.create_directory_structure

# Calibrate intrinsics
python -m posepile.ds.experimental.ntu.main --stage=1
python -m posepile.ds.experimental.ntu.main --calibrate-intrinsics

# Generate pose predictions
for i in {0..114}; do
  SLURM_ARRAY_TASK_ID=$i python -m humcentr_cli.estimate_3d_pose_video \
    --model-path=TODO \
    --video-dir="$DATA_ROOT/ntu/nturgb+d_rgb" \
    --output-dir="$DATA_ROOT/ntu/pred" \
    --file-pattern='**/*.avi' \
    --videos-per-task=100 \
    --no-average-aug \
    --camera-intrinsics-file="$DATA_ROOT/ntu/video_to_camera.pkl"
done

#---------------------------------
# STCN VOS - generate tracked masks to match across time and cameras
# Use this repo https://github.com/isarandi/STCN-buf
# TODO add detailed commands
./ntu_list_videos.py

# Single person
ssubmit -c4 --mem=10G --array=0-317 \
  python -m segment_ntu_or_pku \
  --video-dir="$DATA_ROOT/ntu/nturgb+d_rgb" \
  --output-dir="$DATA_ROOT/ntu/stcn_pred" \
  --file-list-file="$DATA_ROOT/ntu/single_person_videos.txt" \
  --mem-every=20 \
  --mem-size=20 \
  --max-persons=1

# Two people
ssubmit -c4 --mem=10G --array=0-317 \
  python -m segment_ntu_or_pku \
  --video-dir="$DATA_ROOT/ntu/nturgb+d_rgb" \
  --output-dir="$DATA_ROOT/ntu/stcn_pred" \
  --file-list-file="$DATA_ROOT/ntu/two_person_videos.txt" \
  --mem-every=20 \
  --mem-size=20 \
  --max-persons=2

#---------------------------------

# Find out which videos were actually taken from the same view, based on heuristic processing of
# image contents. This is needed because there are sometimes camera movements
# among S00XC00Y videos (for same X and Y), so they need to be calibrated separately.
python -m posepile.ds.experimental.ntu.identify_same_camera_setups

# Triangulate
ssubmit -c4 --gres=gpu:0 --mem=30G --array=0-86 \
  python -m posepile.ds.experimental.ntu.triangulate

# Determine mapping to Kinect pose convention
python -m posepile.ds.experimental.ntu.generate_affine_weights

# Recover global scale
python -m posepile.ds.experimental.ntu.recover_scale

# Generate crops
ssubmit -c4 --gres=gpu:0 --mem=5G --array=0-225 \
  python -m posepile.ds.experimental.ntu.main --stage=2

# num_images=$(find "$DATA_ROOT/ntu_downscaled/" -name '*.jpg' | wc -l)
# should be around 1953691
# Then we segment these downscaled images
# This could be done, but wasn't yet
ssubmit -c4 --mem=5G --array=0-195 \
  python -m humcentr_cli.segment_people \
  --image-root="$DATA_ROOT/ntu_downscaled" \
  --out-dir="$DATA_ROOT/ntu_downscaled/masks"

# Put together and filter
python -m posepile.ds.experimental.ntu.main --stage=3

###########################################################################################

# Generate pose predictions
for i in {0..114}; do
  SLURM_ARRAY_TASK_ID=$i python -m humcentr_cli.estimate_3d_pose_video \
    --model-path=TODO \
    --video-dir="$DATA_ROOT/ntu/nturgb+d_rgb" \
    --output-dir="$DATA_ROOT/ntu/pred_36161de1" \
    --file-pattern='**/*.avi' \
    --videos-per-task=100 \
    --no-average-aug \
    --camera-intrinsics-file="$DATA_ROOT/ntu/video_to_camera.pkl"
done
python -m posepile.ds.experimental.ntu.generate_affine_weights_new
