#!/usr/bin/env bash
# @inproceedings{singleshotmultiperson2018,
#title = {Single-Shot Multi-Person 3D Pose Estimation From Monocular RGB},
#author = {Mehta, Dushyant and Sotnychenko, Oleksandr and Mueller, Franziska and Xu, Weipeng and Sridhar, Srinath and Pons-Moll, Gerard and Theobalt, Christian},
#booktitle = {3D Vision (3DV), 2018 Sixth International Conference on},
#year = {2018},
#}
# http://gvv.mpi-inf.mpg.de/projects/SingleShotMultiPerson
set -euo pipefail
source posepile/functions.sh
check_data_root

# Download and extract the scripts
mkdircd "$DATA_ROOT/muco"
wget http://gvv.mpi-inf.mpg.de/projects/SingleShotMultiPerson/content/muco-3dhp.zip
extractrm muco-3dhp.zip
mv muco-3dhp/* ./
rmdir muco-3dhp

wget http://gvv.mpi-inf.mpg.de/3dhp-dataset/mpi_inf_3dhp.zip
extractrm mpi_inf_3dhp.zip
mv mpi_inf_3dhp/util ./
rm -rf mpi_inf_3dhp

# Verify that the Matlab script files have not been changed
if ! check_md5 mpii_process_multiperson_train_set.m eb99b09d5d97ed04726a7758ac271761; then
  echo "MD5 hash mismatch! Our patch won't work."
  exit 1
fi
if ! check_md5 mpii_create_muco_3dhp_composites.m bb640aebe34d323e6ca1882e76a5e214; then
  echo "MD5 hash mismatch! Our patch won't work."
  exit 1
fi

# Change Windows newlines to Unix
sed -i 's/\r$//' mpii_process_multiperson_train_set.m
sed -i 's/\r$//' mpii_create_muco_3dhp_composites.m

# Apply patches to the Matlab scripts that change the following
# - Chunk ranges (a chunk consists of 500 composites. We can now specify which chunks to process, for parallelizability)
# - Skip already processed chunks
# - Change some hardcoded file paths
# - Better exception handling

patch -e mpii_process_multiperson_train_set.m "$CODE_DIR/ds/muco/mpii_process_multiperson_train_set.patch"
patch -e mpii_create_muco_3dhp_composites.m "$CODE_DIR/ds/muco/mpii_create_muco_3dhp_composites.patch"

# Patches were created with the following commands (*.m_orig already has Unix newlines)
# diff -e mpii_process_multiperson_train_set.m_orig mpii_process_multiperson_train_set.m > mpii_process_multiperson_train_set.patch
# diff -e mpii_create_muco_3dhp_composites.m_orig mpii_create_muco_3dhp_composites.m > mpii_create_muco_3dhp_composites.patch

# Extract all the 3DHP images and masks for all frames and all videos (needs enough free space)
python -m posepile.ds.tdhp.extract_frames_and_masks 1

# Let's now generate the actual MuCo composite images with the Matlab script
# There are 1320 chunks to generate 660000 composites (default in the script, but we will only use 150k)
# The arguments below are the one-based starting and ending chunk indices (inclusive range)
# This can therefore be parallelized by running different chunk ranges on different computers
# On a single machine this will take very long.

"$CODE_DIR/ds/muco/matlab_headless.sh" "mpii_process_multiperson_train_set(1, 1320)"

# Originally detection was run with the https://github.com/isarandi/darknet repo as follows:
# darknet/run_yolo.sh --image-root "$DATA_ROOT/muco" --out-path "$DATA_ROOT/muco/yolov3_detections.pkl"
# The newer version is:
python -m humcentr_cli.detect_people --image-root="$DATA_ROOT/muco" --out-path="$DATA_ROOT/muco/yolov4_detections.pkl"

python -m posepile.ds.muco.postprocess
