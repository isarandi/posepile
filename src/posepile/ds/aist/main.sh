#!/usr/bin/env bash
#@inproceedings{li2021ai,
#  title={Ai choreographer: Music conditioned 3d dance generation with aist++},
#  author={Li, Ruilong and Yang, Shan and Ross, David A and Kanazawa, Angjoo},
#  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
#  pages={13401--13412},
#  year={2021}
#}
#@inproceedings{aist-dance-db,
#   author    = {Shuhei Tsuchida and Satoru Fukayama and Masahiro Hamasaki and Masataka Goto},
#   title     = {AIST Dance Video Database: Multi-genre, Multi-dancer, and Multi-camera Database for Dance Information Processing},
#   booktitle = {Proceedings of the 20th International Society for Music Information Retrieval Conference, {ISMIR} 2019},
#   address   = {Delft, Netherlands},
#   pages     = {501--510},
#   year      = 2019,
#   month     = nov
#}
# https://aistdancedb.ongaaccel.jp/
# https://google.github.io/aistplusplus_dataset/factsfigures.html

set -euo pipefail
source posepile/functions.sh
check_data_root
dataset_name=aist
dataset_dir="$DATA_ROOT/$dataset_name"
mkdircd "$dataset_dir"

# Download the videos
wget https://aistdancedb.ongaaccel.jp/data/video_refined/10M/refined_10M_all_video_url.csv
mkdir -p videos
cat refined_10M_all_video_url.csv | xargs -I{} -P 4 wget --directory-prefix=videos/ {}

# This requires logging in with a Google account, so simple wget doesn't work"
echo "Please download https://storage.cloud.google.com/aist_plusplus_public/20210308/fullset.zip using a browser and save it to $DATA_ROOT/aist/fullset.zip"
extractrm fullset.zip
mv aist_plusplus_final annotations

############################

# Stage 1 picks which examples need to be rendered (adaptive, motion-based sampling)
python -m posepile.ds.aist.main --stage=1

############################

# Stage 2 creates per-person downscaled image crops. Jobs could also be parallelized with e.g. Slurm.
# double check if 12273 is correct (might be different with slighly different detections etc.)
for i in {0..12273}; do
  SLURM_ARRAY_TASK_ID=$i python -m posepile.ds.aist.main --stage=2
done

############################
# Segmentation

# num_images=$(find "$DATA_ROOT/aist_downscaled" -name '*.jpg' | wc -l)
# should be around 5280681
# Then we segment these downscaled images
for i in {0..528}; do
  SLURM_ARRAY_TASK_ID=$i python -m humcentr_cli.segment_people --image-root="$DATA_ROOT/aist_downscaled" --out-dir="$DATA_ROOT/aist_downscaled/masks"
done

############################
python -m posepile.ds.aist.main --stage=3
