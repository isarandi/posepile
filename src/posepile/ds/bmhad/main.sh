#!/usr/bin/env bash
# F. Ofli, R. Chaudhry, G. Kurillo, R. Vidal and R. Bajcsy.
# Berkeley MHAD: A Comprehensive Multimodal Human Action Database. In Proceedings of the IEEE Workshop on Applications on Computer Vision (WACV), 2013.
# https://tele-immersion.citris-uc.org/berkeley_mhad
set -euo pipefail
source posepile/functions.sh
check_data_root

mkdircd "$DATA_ROOT/bmhad"

# TODO Download it here

# Convert to jpeg
python -m posepile.bmhad.pgm_to_jpg

# Generate crops
for i in {0..65}; do
  SLURM_ARRAY_TASK_ID=$i python -m posepile.ds.bmhad.main --stage=1
done

# num_images=$(find "$DATA_ROOT/bmhad_downscaled" -name '*.jpg' | wc -l)
# should be around 527458
# Then we segment these downscaled images
for i in {0..52}; do
  SLURM_ARRAY_TASK_ID=$i python -m humcentr_cli.segment_people --image-root="$DATA_ROOT/bmhad_downscaled" --out-dir="$DATA_ROOT/bmhad_downscaled/masks"
done

# Put all together
python -m posepile.ds.bmhad.main --stage=2

# In the current version only the surface markers are used.
# The joints could be added too, see the bvh_to_joint_positions.py script.