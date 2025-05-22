#!/usr/bin/env bash
# @inproceedings{zou2020detailed,
#  title={3D Human Shape Reconstruction from a Polarization Image},
#  author={Zou, Shihao and Zuo, Xinxin and Qian, Yiming and Wang, Sen and Xu, Chi and Gong, Minglun and Cheng, Li},
#  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
#  year={2020}
#}
# @article{zou2020polarization,
#  title={Polarization Human Shape and Pose Dataset},
#  author={Zou, Shihao and Zuo, Xinxin and Qian, Yiming and Wang, Sen and Guo, Chuan and Xu, Chi and Gong, Minglun and Cheng, Li},
#  journal={arXiv preprint arXiv:2004.14899},
#  year={2020}
#}
# https://jimmyzou.github.io/publication/2020-PHSPDataset
# https://github.com/JimmyZou/PolarHumanPoseShapeDataset
# https://ualbertaca-my.sharepoint.com/:f:/g/personal/szou2_ualberta_ca/EroBwhzfP0NCpl9EdqGeb0kBh6XcZTw1sh2YJ5MJ9PIeMA?e=nIvtdf
# https://drive.google.com/drive/folders/1ZGkpiI99J-4ygD9i3ytJdmyk_hkejKCd?usp=sharing
set -euo pipefail
source posepile/functions.sh
check_data_root

mkdircd "$DATA_ROOT/phps"

extractrm *.tar.gz.0 *.tar.gz *.zip

python -m posepile.tools.find_broken_jpegs "$DATA_ROOT/phps/color" "$DATA_ROOT/phps/ignore_images.txt"
cat "$DATA_ROOT/phps/ignore_images.txt" | while read relpath; do
  rm "$DATA_ROOT/phps/color/$relpath"
done

# ssubmit -c4 --mem=10G --array=0-51
for i in {0..51}; do
  SLURM_ARRAY_TASK_ID=$i python -m humcentr_cli.detect_people \
    --image-root="$DATA_ROOT/phps/color" \
    --out-path="$DATA_ROOT/phps/yolov4_detections.pkl" \
    --images-per-task=10000 \
    --ignore-paths-file="$DATA_ROOT/phps/ignore_images.txt"
done

python -m posepile.tools.pickle_dict_merge \
  "$DATA_ROOT"/phps/yolov4_detections_*.pkl "$DATA_ROOT/phps/yolov4_detections.pkl"

python -m posepile.ds.experimental.phps.main --estimate-up-vectors

# ssubmit -c4 --mem=5G --array=0-141 --gres=gpu:0
for i in {0..141}; do
  SLURM_ARRAY_TASK_ID=$i python -m posepile.ds.experimental.phps.main --stage=1
done

# num_images=$(find "$DATA_ROOT/phps_downscaled/" -name '*.jpg' | wc -l)
# should be around 140479
# Then we segment these downscaled images
for i in {0..14}; do
  SLURM_ARRAY_TASK_ID=$i python -m humcentr_cli.segment_people \
    --image-root="$DATA_ROOT/phps_downscaled" \
    --out-dir="$DATA_ROOT/phps_downscaled/masks"
done

python -m posepile.ds.experimental.phps.main --stage=2
