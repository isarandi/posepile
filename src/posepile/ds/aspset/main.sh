#!/usr/bin/env bash
# @article{nibali2021aspset,
#  title={{ASPset}: An Outdoor Sports Pose Video Dataset With {3D} Keypoint Annotations},
#  author={Nibali, Aiden and Millward, Joshua and He, Zhen and Morgan, Stuart},
#  journal={Image and Vision Computing},
#  pages={104196},
#  year={2021},
#  issn={0262-8856},
#  doi={https://doi.org/10.1016/j.imavis.2021.104196},
#  url={https://www.sciencedirect.com/science/article/pii/S0262885621001013},
#  publisher={Elsevier}
#}
# https://github.com/anibali/aspset-510
# https://archive.org/details/aspset510
set -euo pipefail
source posepile/functions.sh
check_data_root
dataset_name=aspset
dataset_dir="$DATA_ROOT/$dataset_name"

git clone https://github.com/anibali/aspset-510.git "$DATA_ROOT/aspset"
cd "$dataset_dir/src" || exit 1

conda install ezc3d
pip install git+https://github.com/anibali/posekit.git@c9d61d5fa84b3e87dd1a7f0e99ae58aa1e4c759d#egg=posekit
python -m aspset510.bin.download_data --data-dir=../data

python -m posepile.ds.aspset.main --stage=1

# num_images=$(find "$DATA_ROOT/aspset_downscaled" -name '*.jpg' | wc -l)
# should be around 124221
# Then we segment these downscaled images
for i in {0..12}; do
  SLURM_ARRAY_TASK_ID=$i python -m humcentr_cli.segment_people --image-root="$DATA_ROOT/aspset_downscaled" --out-dir="$DATA_ROOT/aspset_downscaled/masks"
done

python -m posepile.ds.aspset.main --stage=2
