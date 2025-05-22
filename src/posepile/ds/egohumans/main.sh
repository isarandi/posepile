#!/usr/bin/env bash
#@article{khirodkar2023egohumans,
#    title={EgoHumans: An Egocentric 3D Multi-Human Benchmark},
#    author={Khirodkar, Rawal and Bansal, Aayush and Ma, Lingni and Newcombe, Richard and Vo, Minh and Kitani, Kris},
#    journal={arXiv preprint arXiv:2305.16487},
#    year={2023}
#  }
# https://rawalkhirodkar.github.io/egohumans/
# https://github.com/rawalkhirodkar/egohumans
# https://arxiv.org/abs/2305.16487

set -euo pipefail
source posepile/functions.sh
check_data_root

dataset_name=egohumans
dataset_dir="$DATA_ROOT/$dataset_name"
mkdircd "$dataset_dir"

# https://github.com/glotlabs/gdrive
# gdrive account add
gdrive files download --recursive 1JD963urzuzV_R_6FOVOtlx8UupwUuknR

for dir in 0*_*; do
  pushd "$dir"
  for n in *.tar.gz; do
    tar xvf "$n" --strip-components=8 && rm "$n"
  done
  popd
done

find . -name '*.jpg' > image_paths.txt
grep -v '\(left\|right\|processed_data\|undistorted\)' image_paths.txt > image_paths_rgb.txt

grep '/ego/' image_paths_rgb.txt > image_paths_rgb_ego.txt
grep 'volleyball/exo/cam02/' image_paths_rgb.txt > image_paths_rgb_exo_1080.txt
grep '/exo/' image_paths_rgb.txt | grep -v 'volleyball/exo/cam02/' | grep -v 'undistorted' > image_paths_rgb_exo_2160.txt

# Handle each resolution separately, so they can be batched
python -m humcentr_cli.detect_people --image-root="$dataset_dir" --image-paths-file="$dataset_dir/image_paths_rgb_ego.txt" --out-path="$dataset_dir/yolov4_detections_ego.pkl" --batch-size=8
python -m humcentr_cli.detect_people --image-root="$dataset_dir" --image-paths-file="$dataset_dir/image_paths_rgb_exo_2160.txt" --out-path="$dataset_dir/yolov4_detections_exo_2160.pkl" --batch-size=8
python -m humcentr_cli.detect_people --image-root="$dataset_dir" --image-paths-file="$dataset_dir/image_paths_rgb_exo_1080.txt" --out-path="$dataset_dir/yolov4_detections_exo_1080.pkl" --batch-size=8

python -m posepile.tools.pickle_dict_merge "$dataset_dir"/yolov4_detections_*.pkl "$dataset_dir/yolov4_detections.pkl"