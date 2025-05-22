#!/usr/bin/env bash
#@inproceedings { PonBaa2010a,
#  author = {Gerard Pons-Moll and Andreas Baak and Thomas Helten and Meinard M{\"u}ller and Hans-Peter Seidel and Bodo Rosenhahn},
#  title = {Multisensor-Fusion for 3D Full-Body Human Motion Capture},
#  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
#  year = {2010},
#  doi = {10.1109/CVPR.2010.5540153},
#}
# @inproceedings { BaaHel2010a,
#  author = {Andreas Baak and Thomas Helten and Meinard M{\"u}ller and Gerard Pons-Moll and Bodo Rosenhahn and Hans-Peter Seidel},
#  title = {Analyzing and Evaluating Markerless Motion Tracking Using Inertial Sensors },
#  booktitle = {European Conference on Computer Vision (ECCV Workshops)},
#  year = {2010},
#  doi = {10.1007/978-3-642-35749-7_11},
#}
# http://www.tnt.uni-hannover.de/project/MPI08_Database/
set -euo pipefail
source posepile/functions.sh
check_data_root

mkdircd "$DATA_ROOT/mpi08"

url_base=http://www.tnt.uni-hannover.de/project/MPI08_Database/
wget $url_base/ab.tar.gz
wget $url_base/br.tar.gz
wget $url_base/hb.tar.gz
wget $url_base/mm.tar.gz
wget $url_base/InputFiles.tar.gz
wget $url_base/Documents.tar.gz
wget $url_base/MPI08_PriorFiles.tar.gz
for n in *.tar.gz; do tar xf "$n"; rm "$n"; done

find . -name '*99.avi' -delete

python -m posepile.ds.experimental.mpi08.save_camconfig

ssubmit -c4 --array=0-81 python -m humcentr_cli.estimate_3d_pose_video --video-dir="$DATA_ROOT/mpi08" --output-dir="$DATA_ROOT/mpi08/pred" --file-pattern='**/*.avi' --videos-per-task=10 --camera-file="$DATA_ROOT/mpi08/cameras.pkl" --model-path=/nodes/brewdog/work3/sarandi/data_reprod//experiments/kerasreprod/effv2l_ghost_each_new_aist_3e-4_2gpu_/model_multi_v1.2_distnew --no-average-aug

ssubmit -c8 --mem=10G --gres=gpu:0 --array=0-101 python -m posepile.ds.experimental.mpi08_prepare.triangulate_mpi08
ssubmit -c8 --mem=10G --gres=gpu:0 --array=0-101 python -m posepile.ds.mpi08.main --stage=1

# num_images=$(find "$DATA_ROOT/mpi08_downscaled/" -name '*.jpg' | wc -l)
# should be around 165379
# Then we segment these downscaled images
ssubmit -c4 --array=0-16 python -m humcentr_cli.segment_people --image-root="$DATA_ROOT/mpi08_downscaled" --out-dir="$DATA_ROOT/mpi08_downscaled/masks"

python -m posepile.ds.experimental.mpi08.main --stage=2

url_base=http://www.tnt.uni-hannover.de/papers/data/901/
mkdir 901
cd 901
wget $url_base/ab.tar.gz
wget $url_base/hb.tar.gz
wget $url_base/InputFiles.tar.gz
wget $url_base/PriorFiles.tar.gz
wget $url_base/Documents.tar.gz
for n in *.tar.gz; do tar xf "$n"; rm "$n"; done