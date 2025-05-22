#!/usr/bin/env bash
# Herve Jegou, Matthijs Douze and Cordelia Schmid
# "Hamming Embedding and Weak geometry consistency for large scale image search"
# Proceedings of the 10th European conference on Computer vision, October, 2008
# https://lear.inrialpes.fr/~jegou/data.php
set -euo pipefail
source posepile/functions.sh
check_data_root

mkdircd "$DATA_ROOT/inria_holidays"

for i in 1 2; do
  wget ftp://ftp.inrialpes.fr/pub/lear/douze/data/jpg$i.tar.gz
  tar -xvf jpg$i.tar.gz
  rm jpg$i.tar.gz
done

python -m posepile.ds.inria_holidays.prepare_images

# Originally detection was run with the https://github.com/isarandi/darknet repo as follows:
# darknet/run_yolo.sh --image-root "$DATA_ROOT/inria_holidays/jpg_small" --out-path "$DATA_ROOT/inria_holidays/yolov3_person_detections.pkl"
# The newer version is:
python -m humcentr_cli.detect_people --image-root="$DATA_ROOT/inria_holidays/jpg_small" --out-path="$DATA_ROOT/inria_holidays/yolov4_person_detections.pkl"

python -m posepile.ds.inria_holidays.find_nonperson_images
