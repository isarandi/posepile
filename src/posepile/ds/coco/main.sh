#!/usr/bin/env bash
# @inproceedings{lin2014coco,
#   title={Microsoft COCO: Common Objects in Context},
#   author={Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Doll{\'a}r, Piotr and Zitnick, C Lawrence},
#   booktitle={ECCV},
#   year={2014}
# }
# https://cocodataset.org/#download

set -euo pipefail
source posepile/functions.sh
check_data_root

dataset_name=coco
dataset_dir="$DATA_ROOT/$dataset_name"
mkdircd "$dataset_dir"

for year in 2014 2017; do
  for phase in train val test; do
    wget "http://images.cocodataset.org/zips/${phase}${year}.zip"
    extractrm "${phase}${year}.zip"
  wget "http://images.cocodataset.org/annotations/annotations_trainval${year}.zip"
  extractrm "annotations_trainval${year}.zip"
done




# Download COCO Wholebody from https://github.com/jin-s13/COCO-WholeBody
# pip install xtcocotools
python -m posepile.ds.coco.main --wholebody