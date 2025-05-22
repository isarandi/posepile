#!/usr/bin/env bash

# Steps I did with a 2020 MeTRAbs model (ResNet101 or 152 backbone)
# - Predict poses per camera
# - Triangulate
# - Recalibrate extrinsics for some of the cameras
# (the original calibrations from the dataset are not quite accurate for some of the camears)
#-------
# Later, with the WACV23 MeTRAbs-EffV2L-384 model I did
# - Undistort videos first
# - Predict poses per camera
# - Triangulate using the recalibrated extrinsics

sbatch -c4 --mem=6G --array=0-174 \
  python -m humcentr_cli.estimate_3d_pose_video \
  --model-path=$MODEL_PATH \
  --video-dir=$DATA_ROOT/panoptic \
  --file-pattern='*_dance*/hdVideos/*_undistorted.mp4,*_moonbaby*/kinectVideos/*_undistorted.mp4' \
  --output-dir="$DATA_ROOT/panoptic-more/pred_36161de1_aug" \
  --batch-size=6 \
  --internal-batch-size=40 \
  --num-aug=5 \
  --videos-per-task=1 \
  --skeleton='' \
  --camera-file=$DATA_ROOT/panoptic/dance_cameras.pkl \
  --no-average-aug

python -m posepile.ds.panoptic.dance.triangulate \
  --skeleton-types-file="$DATA_ROOT/skeleton_conversion/skeleton_types_huge8.pkl" \
  --skeleton='' \
  --detector-flip-aug \
  --output-path="$DATA_ROOT/panoptic-more/pred_36161de1_aug"
