#!/usr/bin/env bash
# @inproceedings{Huang:CVPR:2022,
#title = {Capturing and Inferring Dense Full-Body Human-Scene Contact},
#author = {Huang, Chun-Hao P. and Yi, Hongwei and H{\"o}schle, Markus and Safroshkin, Matvey and Alexiadis, Tsvetelina and Polikovsky, Senya and Scharstein, Daniel and Black, Michael J.},
#  booktitle = {Proceedings IEEE/CVF Conf.~on Computer Vision and Pattern Recognition (CVPR)},
#  year = {2022},
#}
# https://rich.is.tue.mpg.de/
set -euo pipefail
source posepile/functions.sh
check_data_root

mkdircd "$DATA_ROOT/rich"
# From https://rich.is.tue.mpg.de/download.php download spec-syn.zip and spec-mtp.zip

# download the download_{train,val,test}.sh scripts into the rich dir
# download the scan_calibration.zip, {train,val,test}_body.zip files into the rich dir

read -rp 'Username: ' username
read -rsp 'Password: ' password

for phase in train val test; do
  sed -i 's/\r$//' download_$phase.sh
  grep -Po "(?<=-O ')[^']+" download_$phase.sh > ${phase}_files.txt

  cat ${phase}_files.txt | while read name; do
    python -m posepile.ds.rich.download_as_jpeg --source-file "$phase/$name" --username="$username" --password="$password"
  done
done

extractrm scan_calibration.zip
extractrm train_body.zip
extractrm val_body.zip
extractrm test_body.zip
mv ps/project/multi-ioi/rich_release/* ./
rm -rf ps

python -m humcentr_cli.detect_people --image-root="$DATA_ROOT/rich" --out-path="$DATA_ROOT/rich/yolov4_detections.pkl" --file-pattern='**/*.jpg'
# python -m posepile.tools.pickle_dict_merge "$DATA_ROOT"/rich/yolov4_detections_*.pkl "$DATA_ROOT/rich/yolov4_detections.pkl"
# rm "$DATA_ROOT"/rich/yolov4_detections_*.pkl

# clone https://github.com/vchoutas/smplx.git and add it to the PYTHONPATH
export PYTHONPATH=$somepath/smplx
for i in {0..88}; do
  SLURM_ARRAY_TASK_ID=$i python -m posepile.ds.rich.main --stage=1
done

# num_images=$(find "$DATA_ROOT/rich_downscaled/" -name '*.jpg' | wc -l)
# should be around 96146
# Then we segment these downscaled images, needs GPUs
for i in {0..9}; do
  SLURM_ARRAY_TASK_ID=$i python -m humcentr_cli.segment_people --image-root="$DATA_ROOT/rich_downscaled" --out-dir="$DATA_ROOT/rich_downscaled/masks"
done

python -m posepile.ds.rich.main --stage=2

# Convert the test set jpegs to videos
find "$DATA_ROOT/rich/test" -type d | while read -r folder; do
  if ls "$folder"/*.jpeg &>/dev/null; then
    echo "Processing folder: $folder"
    ffmpeg -nostdin -loglevel quiet -y -framerate 30 -pattern_type glob -i "$folder/*.jpeg" \
      -vf "scale=min(4096\,iw):-2" \
      -c:v h264_nvenc -rc constqp -qp 21 -preset p7 -tune hq -pix_fmt yuv420p "${folder}.mp4"
  fi
done