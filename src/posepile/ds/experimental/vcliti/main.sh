#!/usr/bin/env bash
# @article{alexiadis2017integrated,
#  title={An Integrated Platform for Live 3D Human Reconstruction and Motion Capturing},
#  author={Alexiadis, Dimitrios S and Chatzitofis, Anargyros and Zioulis, Nikolaos and Zoidi, Olga and Louizis,
#            Georgios and Zarpalas, Dimitrios and Daras, Petros},
#  journal={IEEE Transactions on Circuits and Systems for Video Technology},
#  volume={27},
#  number={4},
#  pages={798--813},
#  year={2017},
#  publisher={IEEE}
#}
# https://vcl.iti.gr/dataset/dataset-of-multiple-kinect2-rgb-d-streams/
# https://vcl.iti.gr/dataset/datasets-of-multiple-kinect2-rgb-d-streams-and-skeleton-tracking/
set -euo pipefail
source posepile/functions.sh
check_data_root

mkdircd "$DATA_ROOT/vcliti"
# Download dataset here
# TODO add commands

python -m posepile.ds.experimental.vcliti.unpack
python -m posepile.ds.experimental.vcliti.save_camconfig

ssubmit -c4 --array=0-9 \
  python -m humcentr_cli.estimate_3d_pose \
  --model-path=/nodes/brewdog/work3/sarandi/data_reprod//experiments/kerasreprod/effv2l_ghost_each_new_aist_3e-4_2gpu_/model_multi_v1.2_distnew \
  --image-root="$DATA_ROOT/vcliti" \
  --file-pattern='**/*rgb*.jpg' \
  --out-path="$DATA_ROOT/vcliti/pred/metrabs_pred" \
  --camera-file="$DATA_ROOT/vcliti/cameras.pkl" \
  --no-suppress-implausible-poses \
  --detector-threshold=0.1 \
  --no-average-aug \
  --horizontal-flip

python -m posepile.tools.pickle_dict_merge "$DATA_ROOT/vcliti/pred/"metrabs_pred* "$DATA_ROOT/vcliti/metrabs_pred.pkl"
rm -rf "$DATA_ROOT/vcliti/pred"

ssubmit -c8 --mem=10G --gres=gpu:0 --array=0-21 \
  python -m posepile.ds.experimental.vcliti.triangulate
ssubmit -c4 --mem=8G --gres=gpu:0 --array=0-21 \
  python -m posepile.ds.experimental.vcliti.main --stage=1

# num_images=$(find "$DATA_ROOT/vcliti_downscaled/" -name '*.jpg' | wc -l)
# should be around 19173
# Then we segment these downscaled images
ssubmit -c4 --array=0-1 \
  python -m humcentr_cli.segment_people \
  --image-root="$DATA_ROOT/vcliti_downscaled" \
  --out-dir="$DATA_ROOT/vcliti_downscaled/masks"

python -m posepile.ds.experimental.vcliti.main --stage=2
