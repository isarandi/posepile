#!/usr/bin/env bash
# @inproceedings{sampieri2022pose,
#  title={Pose Forecasting in Industrial Human-Robot Collaboration},
#  author={Sampieri, Alessio and di Melendugno, Guido Maria D'Amely and Avogaro, Andrea and Cunico, Federico and Setti, Francesco and Skenderi, Geri and Cristani, Marco and Galasso, Fabio},
#  booktitle={European Conference on Computer Vision},
#  year={2022},
#}
# https://github.com/federicocunico/human-robot-collaboration
# https://github.com/AlessioSam/CHICO-PoseForecasting/
# https://doi.org/10.1007/978-3-031-19839-7_4
# https://arxiv.org/abs/2208.07308

set -euo pipefail
source posepile/functions.sh
check_data_root

mkdircd "$DATA_ROOT/chico"

# Download the zip files into the chico directory (https://github.com/federicocunico/human-robot-collaboration)
# Then:

for name in *.zip; do
  extractrm "$name"
done

# Delete some redundant files
find . -name C5.mp4 -delete
find . -name C6.mp4 -delete
find . -name C9.mp4 -delete

python -m posepile.ds.experimental.chico.save_camconfig

# Generate pose predictions
python -m humcentr_cli.estimate_3d_pose_video --model-path=https://bit.ly/metrabs_l --video-dir="$DATA_ROOT/chico/" --file-pattern='**/*.mp4' --skeleton='' --output-dir=$DATA_ROOT/chico/1c6f6193_pred --internal-batch-size=10 --num-aug=5  --suppress-implausible-poses --videos-per-task=10000 --no-high-quality-viz --detector-flip-aug --antialias-factor=2 --batch-size=6 --internal-batch-size=40 --camera-file=$DATA_ROOT/chico/cameras_only_intrinsics.pkl --max-detections=1 --no-average-aug

python -m posepile.ds.experimental.chico.calibrate_extrinsics

python -m posepile.ds.experimental.chico.visualize