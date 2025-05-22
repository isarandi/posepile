#!/usr/bin/env bash
#@inproceedings{SPEC:ICCV:2021,
#  title = {{SPEC}: Seeing People in the Wild with an Estimated Camera},
#  author = {Kocabas, Muhammed and Huang, Chun-Hao P. and Tesch, Joachim and M\"uller, Lea and Hilliges, Otmar and Black, Michael J.},
#  booktitle = {Proc. International Conference on Computer Vision (ICCV)},
#  year = {2021},
#}
# https://spec.is.tue.mpg.de/
# https://github.com/mkocabas/SPEC
set -euo pipefail
source posepile/functions.sh
check_data_root

mkdircd "$DATA_ROOT/spec"
# From https://spec.is.tue.mpg.de/download.php download spec-syn.zip and spec-mtp.zip

extractrm spec-syn.zip
extractrm spec-mtp.zip

python -m humcentr_cli.detect_people --image-root="$DATA_ROOT/spec/spec-syn" --file-pattern='**/*.png' \
  --out-path="$DATA_ROOT/spec/yolov4_detections.pkl" --image-type=png

python -m posepile.ds.spec.main