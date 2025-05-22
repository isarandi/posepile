#!/usr/bin/env bash
#@inproceedings{vendrow2023jrdb,
#  title={{JRDB-Pose}: A large-scale dataset for multi-person pose estimation and tracking},
#  author={Vendrow, Edward and Le, Duy Tho and Cai, Jianfei and Rezatofighi, Hamid},
#  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
#  pages={4811--4820},
#  year={2023}
#}
#@article{MartinMartin21PAMI,
#  title         = {{JRDB}: A dataset and benchmark of egocentric robot visual perception of humans in built environments},
#  author        = {Mart\'in-Mart\'in, Roberto and Patel, Mihir and Rezatofighi, Hamid and
#                   Shenoi, Abhijeet and Gwak, JunYoung and Frankel, Eric and Sadeghian, Amir and
#                   Savarese, Silvio},
#  journal       = TPAMI,
#  year          = {2021},
#  doi           = {10.1109/TPAMI.2021.3070543},
#  note          = {Early access}
#}
# https://jrdb.erc.monash.edu/
set -euo pipefail
source posepile/functions.sh
check_data_root

mkdircd "$DATA_ROOT/jrdb"

python -m humcentr_cli.detect_people --image-root="$DATA_ROOT/jrdb/train_dataset/images" --out-path="$DATA_ROOT/jrdb/train_dataset/yolov4_detections_distorted.pkl" --file-pattern='**/*.jpg'
python -m posepile.ds.jrdb.main
