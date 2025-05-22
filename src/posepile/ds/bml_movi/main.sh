#!/usr/bin/env bash
#  @data{SP2/JRHDRN_2020,
#  author = {Ghorbani, Saeed and Mahdaviani, Kimia and Anne Thaler and Konrad Kording and Douglas James Cook and Gunnar Blohm and Nikolaus F. Troje},
#  publisher = {Scholars Portal Dataverse},
#  title = {{MoVi: A Large Multipurpose Motion and Video Dataset}},
#  year = {2020},
#  version = {V5},
#  doi = {10.5683/SP2/JRHDRN},
#  url = {https://doi.org/10.5683/SP2/JRHDRN}
#  }
set -euo pipefail
source posepile/functions.sh
check_data_root

mkdircd "$DATA_ROOT/bml_movi"

# Download all 1057 files from https://doi.org/10.5683/SP2/JRHDRN into "$DATA_ROOT/bml_movi"

extractrm *.tar *.zip

git clone https://github.com/saeed1262/MoVi-Toolbox.git

# There are 1044 videos in the dataset  (208 = ceil(1044/5)-1)
# Generate pose predictions
for i in {0..208}; do
  SLURM_ARRAY_TASK_ID=$i python -m humcentr_cli.estimate_3d_pose_video \
    --model-path=TODO \
    --video-dir="$DATA_ROOT/bml_movi" \
    --output-dir="$DATA_ROOT/bml_movi/pred" \
    --file-pattern='*.mp4,*.avi' \
    --videos-per-task=5
done

# Fit affine mapping from the pose estimator's skeleton to the BML-MoVi skeleton
python -m posepile.ds.bml_movi.main --generate-affine-weights

# Calibrate the cameras using the affine mapped predictions
CUDA_VISIBLE_DEVICES='' python -m posepile.ds.bml_movi.main --calibrate-cameras

# Detect people in videos
for i in {0..208}; do
  SLURM_ARRAY_TASK_ID=$i python -m humcentr_cli.detect_people_video \
    --video-dir="$DATA_ROOT/bml_movi" \
    --output-dir="$DATA_ROOT/bml_movi/detections" \
    --file-pattern='*.mp4,*.avi' \
    --videos-per-task=5
done

# Generate crops
#  -c2 --mem=10G --gres=gpu:0 --array=0-1043
for i in {0..1043}; do
  SLURM_ARRAY_TASK_ID=$i python -m posepile.ds.bml_movi.main --stage=1
done

# num_images=$(find "$DATA_ROOT/bml_movi_downscaled/" -name '*.jpg' | wc -l)
# should be around 562691
# Then we segment these downscaled images
for i in {0..56}; do
  SLURM_ARRAY_TASK_ID=$i python -m humcentr_cli.segment_people --image-root="$DATA_ROOT/bml_movi_downscaled" --out-dir="$DATA_ROOT/bml_movi_downscaled/masks"
done

# Put all together
python -m posepile.ds.bml_movi.main --stage=2
