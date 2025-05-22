#!/usr/bin/env bash
#@inproceedings{pumarola20193dpeople,
#    title={{3DPeople: Modeling the Geometry of Dressed Humans}},
#    author={Pumarola, Albert and Sanchez, Jordi and Choi, Gary and Sanfeliu, Alberto and Moreno-Noguer, Francesc},
#    booktitle={International Conference on Computer Vision (ICCV)},
#    year={2019}
#}
# https://cv.iri.upc-csic.es/
# https://www.albertpumarola.com/research/3DPeople/index.html
# https://github.com/albertpumarola/3DPeople-Dataset
set -euo pipefail
source posepile/functions.sh
check_data_root

mkdircd "$DATA_ROOT/3dpeople"

cat urls | xargs -I{} -P 4 wget {}

for name in *.tar.gz; do
  tar xf "$name"
  rm "$name"
done

python -m posepile.ds.tdpeople.main
python -m posepile.ds.tdpeople.main --composite --stage=1
python -m humcentr_cli.detect_people --image-root="$DATA_ROOT/muco_3dpeople/images" --out-path="$DATA_ROOT/muco_3dpeople/yolov4_detections.pkl"
python -m posepile.ds.tdpeople.main --composite --stage=2