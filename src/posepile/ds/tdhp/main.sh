#!/usr/bin/env bash
#@inproceedings{mono-3dhp2017,
# author = {Mehta, Dushyant and Rhodin, Helge and Casas, Dan and Fua, Pascal and Sotnychenko, Oleksandr and Xu, Weipeng and Theobalt, Christian},
# title = {Monocular 3D Human Pose Estimation In The Wild Using Improved CNN Supervision},
# booktitle = {3D Vision (3DV), 2017 Fifth International Conference on},
# url = {http://gvv.mpi-inf.mpg.de/3dhp_dataset},
# year = {2017},
# organization={IEEE},
# doi={10.1109/3dv.2017.00064},
#}
# http://gvv.mpi-inf.mpg.de/3dhp-dataset/
set -euo pipefail
source posepile/functions.sh
check_data_root

cd "$DATA_ROOT"
wget http://gvv.mpi-inf.mpg.de/3dhp-dataset/mpi_inf_3dhp.zip
extractrm mpi_inf_3dhp.zip
mv mpi_inf_3dhp 3dhp
cd 3dhp

sed -i 's/subjects=(1 2)/subjects=(1 2 3 4 5 6 7 8)/' conf.ig
sed -i "s/destination='.\/'/'destination=$DATA_ROOT\/3dhp\/'/" conf.ig
sed -i "s/ready_to_download=0/ready_to_download=1/" conf.ig

bash get_dataset.sh
bash get_testset.sh

mv mpi_inf_3dhp_test_set/mpi_inf_3dhp_test_set/TS* ./
mv mpi_inf_3dhp_test_set/mpi_inf_3dhp_test_set/test_util ./
mv mpi_inf_3dhp_test_set/mpi_inf_3dhp_test_set/README.txt ./README_testset.txt
rmdir mpi_inf_3dhp_test_set/mpi_inf_3dhp_test_set
rmdir mpi_inf_3dhp_test_set

python -m posepile.tdhp.extract_frames_and_masks_3dhp 5
python -m posepile.tdhp.find_images_for_detection > 3dhp_images_for_detection.txt

# Originally detection was run with the https://github.com/isarandi/darknet repo as follows:
# darknet/run_yolo.sh --image-paths-file 3dhp_images_for_detection.txt --out-path "$DATA_ROOT/3dhp/yolov3_person_detections.pkl"
# The newer version is:
python -m humcentr_cli.detect_people --image-paths-file=3dhp_images_for_detection.txt --out-path="$DATA_ROOT/3dhp/yolov3_person_detections.pkl"
rm 3dhp_images_for_detection.txt

# Now create the "Full" version as well.
# This "full" version uses all cameras (not just chest height), all 28 joints,
# segmentation from the usual segmentation model
# of this project instead of the chroma masks, TF2 YOLOv4 instead of darknet YOLOv3,
# no pre-extracted frames, but on-the-fly reading of the videos and on-the-fly detection,
# frame sampling is done purely based on motion (at least 100 mm) instead of first downsampling
# by a factor of 5. No filtering of truncated poses.
ln -s 3dhp 3dhp_full

# Generate crops
for i in {0..223}; do
  SLURM_ARRAY_TASK_ID=$i python -m posepile.tdhp.full --stage=1
done

python -m posepile.tdhp.full --stage=2