#!/usr/bin/env bash
#@article{chatzitofis2020human4d,
#  title={HUMAN4D: A Human-Centric Multimodal Dataset for Motions and Immersive Media},
#  author={Chatzitofis, Anargyros and Saroglou, Leonidas and Boutis, Prodromos and Drakoulis, Petros and Zioulis, Nikolaos and Subramanyam, Shishir and Kevelham, Bart and Charbonnier, Caecilia and Cesar, Pablo and Zarpalas, Dimitrios and others},
#  journal={IEEE Access},
#  volume={8},
#  pages={176241--176262},
#  year={2020},
#  publisher={IEEE}
#}
# https://tofis.github.io/human4d_dataset/
set -euo pipefail
source posepile/functions.sh
check_data_root

mkdircd "$DATA_ROOT/human4d"

# Download it such that the frames have paths like
# $DATA_ROOT/human4d/S2/19-07-12-09-09-06/Dump/color/438_M72h_color_46613.png

python -m humcentr_cli.detect_people \
  --image-root="$DATA_ROOT/human4d" \
  --file-pattern='**/*.png' \
  --out-path="$DATA_ROOT/human4d/yolov4_detections.pkl" \
  --image-type=png \
  --rot=270

python -m posepile.ds.human4d.main --stage=1

# num_images=$(find "$DATA_ROOT/human4d_downscaled/" -name '*.jpg' | wc -l)
# should be around 26913
# Then we segment these downscaled images
for i in {0..3}; do
  SLURM_ARRAY_TASK_ID=$i python -m humcentr_cli.segment_people \
    --image-root="$DATA_ROOT/human4d_downscaled" \
    --out-dir="$DATA_ROOT/human4d_downscaled/masks"
done
