#!/usr/bin/env bash
# @inproceedings{HICV11:UMPM,
#  author = {Aa, N.P.~van~der and Luo, X. and Giezeman, G.J. and Tan, R.T. and Veltkamp, R.C.},
#  year = 2011,
#  title = {Utrecht Multi-Person Motion (UMPM) benchmark: a multi-person dataset with
#    synchronized video and motion capture data for evaluation of articulated
#    human motion and interaction},
#  booktitle= { Proceedings of the Workshop on Human Interaction in Computer Vision (HICV), in
#      conjunction with ICCV 2011}
#}
# https://www2.projects.science.uu.nl/umpm/
set -euo pipefail
source posepile/functions.sh
check_data_root

mkdircd "$DATA_ROOT/umpm"

read -rp 'Username: ' user
read -rsp 'Password: ' password

wget https://www2.projects.science.uu.nl/umpm/data/urls.txt
cat urls.txt | sed $'s/\r$//' | xargs -I{} -P 16 wget --http-user="$user" --http-passwd="$password" {}

for name in *.zip; do extractrm "$name" & done

cd Video
# This step takes long, we extract about 1 TB of video data here
for name in *.xz; do xz --decompress "$name" & done

# Uncompressed video is huge, so let us extract the individual frames as jpeg files and thereby
# compress it by a factor of 10
for i in {0..99}; do
  SLURM_ARRAY_TASK_ID=$i python -m posepile.ds.umpm.extract_frames
done

rm "$DATA_ROOT/umpm/Video/*.avi"

# Fix the JSON files (trailing comma in list not allowed by JSON standard)
for jsonfile in "$DATA_ROOT/umpm/"*.json; do
  sed -z 's/,\n}/\n}/' -i "$jsonfile"
done

# Replace bad JSON file
mv p2_free_1.json{,_broken}
mv p2_free_2.json{,_broken}
mv p3_circle_2.json{,_broken}
echo '{
  "id":"p2_free_1",
  "bg":"backgr_n27_1",
  "calib":"calib_n27",
  "video_dir":"Video",
  "gt_dir":"Groundtruth",
  "bg_dir":"Background",
  "calib_dir":"Calib"
}' >p2_free_1.json

echo '{
  "id":"p3_circle_2",
  "bg":"backgr_l06_7",
  "calib":"calib_l06",
  "video_dir":"Video",
  "gt_dir":"Groundtruth",
  "bg_dir":"Background",
  "calib_dir":"Calib"
}' >p3_circle_2.json

for i in {0..8}; do
  SLURM_ARRAY_TASK_ID=$i python -m humcentr_cli.detect_people \
    --images-per-task=100000 \
    --image-root="$DATA_ROOT/umpm" \
    --out-path="$DATA_ROOT/umpm/yolov4_detections.pkl"
done

python posepile.tools.pickle_dict_merge "$DATA_ROOT"/umpm/yolov4_detections_*.pkl "$DATA_ROOT/umpm/yolov4_detections.pkl"
rm "$DATA_ROOT"/umpm/yolov4_detections_*.pkl

# Generate crops
for i in {0..73}; do
  SLURM_ARRAY_TASK_ID=$i python -m posepile.ds.umpm.main --stage=1
done

# num_images=$(find "$DATA_ROOT/umpm_downscaled/" -name '*.jpg' | wc -l)
# should be around 166833
# Then we segment these downscaled images
for i in {0..16}; do
  SLURM_ARRAY_TASK_ID=$i python -m humcentr_cli.segment_people --image-root="$DATA_ROOT/umpm_downscaled" --out-dir="$DATA_ROOT/umpm_downscaled/masks"
done

# Put all together
python -m posepile.ds.umpm.main --stage=2
