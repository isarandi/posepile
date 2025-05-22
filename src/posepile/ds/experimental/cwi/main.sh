#!/usr/bin/env bash
# @inproceedings{10.1145/3458305.3478452,
#author = {Reimat, Ignacio and Alexiou, Evangelos and Jansen, Jack and Viola, Irene and Subramanyam, Shishir and Cesar, Pablo},
#title = {CWIPC-SXR: Point Cloud Dynamic Human Dataset for Social XR},
#year = {2021},
#doi = {10.1145/3458305.3478452},
#booktitle = {Proceedings of the 12th ACM Multimedia Systems Conference},
#pages = {300–306},
#}
# https://www.dis.cwi.nl/cwipc-sxr-dataset/
set -euo pipefail
source posepile/functions.sh
check_data_root

mkdircd "$DATA_ROOT/cwi"

# Check https://www.dis.cwi.nl/cwipc-sxr-dataset/ for the up-to-date URL
wget http://192.16.197.145:8086/dataset_hierarchy.tgz
tar xvf dataset_hierarchy.tgz
mv cwipc-sxr-dataset/* ./
rmdir cwipc-sxr-dataset

find . -name download.sh | while read script; do
  pushd "$(dirname $script)"
  grep raw_files download.sh | tr -d "\r" | bash
  popd
done

find . -name '*.mkv' | while read x; do
  mkvextract attachments "$x" "1:${x}_calibration.json"
done

python -m posepile.ds.experimental.cwi.save_camconfig

for i in {0..32}; do
  SLURM_ARRAY_TASK_ID=$i python -m humcentr_cli.estimate_3d_pose_video \
    --video-dir="$DATA_ROOT/cwi" \
    --output-dir="$DATA_ROOT/cwi/pred" \
    --file-pattern='**/*.mkv' \
    --videos-per-task=10 \
    --camera-file="$DATA_ROOT/cwi/cameras.pkl" \
    --model-path=TODO
done

# ssubmit -c4 --mem=10G --array=0-51
ssubmit -c8 --mem=10G --gres=gpu:0 --array=0-45 \
  python -m posepile.ds.experimental.cwi.triangulate

ssubmit -c8 --mem=10G --gres=gpu:0 --array=0-45 \
  python -m posepile.ds.experimental.cwi.main --stage=1

# num_images=$(find "$DATA_ROOT/cwi_downscaled/" -name '*.jpg' | wc -l)
# should be around 67362
# Then we segment these downscaled images
for i in {0..6}; do
  SLURM_ARRAY_TASK_ID=$i python -m humcentr_cli.segment_people \
    --image-root="$DATA_ROOT/cwi_downscaled" \
    --out-dir="$DATA_ROOT/cwi_downscaled/masks"
done

python -m posepile.ds.experimental.cwi.main --stage=2
