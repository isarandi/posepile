#!/usr/bin/env bash
# @article{wang2023freeman,
#  title={FreeMan: Towards Benchmarking 3D Human Pose Estimation in the Wild},
#  author={Wang, Jiong and Yang, Fengyu and Gou, Wenbo and Li, Bingliang and Yan, Danqi and Zeng, Ailing and Gao, Yijun and Wang, Junle and Zhang, Ruimao},
#  journal={arXiv preprint arXiv:2309.05073},
#  year={2023}
#}
# https://wangjiongw.github.io/freeman/
# https://github.com/wangjiongw/FreeMan_API
# https://huggingface.co/datasets/wjwow/FreeMan

set -euo pipefail
source posepile/functions.sh
check_data_root
dataset_name=freeman
dataset_dir="$DATA_ROOT/$dataset_name"
mkdircd "$dataset_dir"

# Get an API token from HuggingFace
token="YOUR TOKEN HERE"
get_file(){
  local filepath=$1
  mkdir -p "$(dirname "$p")"
  wget --header="Authorization: Bearer $token" "https://huggingface.co/datasets/wjwow/FreeMan/resolve/main/$filepath" -O "$filepath" --continue
}

mkdir videos

cat "$CODE_DIR/ds/freeman/file_list" | while read p; do
  if [[ ! -f "$p" ]]; then
    get_file "$p"
  fi
done

for name in keypoints3d cameras motions; do
  unzip $name.zip -d $name/ && rm $name.zip
done

cd videos
extractrm *.zip

python -m humcentr_cli.detect_people_video --video-dir="$dataset_dir" --output-dir="$dataset_dir/detections"
for i in {0..560}; do
  SLURM_ARRAY_TASK_ID=$i python -m posepile.ds.freeman.main --stage=1
done
