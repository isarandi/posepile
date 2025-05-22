#!/usr/bin/env bash
# @article{liu2021neural,
# title={Neural Actor: Neural Free-view Synthesis of Human Actors with Pose Control},
# author={Lingjie Liu and Marc Habermann and Viktor Rudnev and Kripasindhu Sarkar and Jiatao Gu and Christian Theobalt},
# year={2021},
# journal = {ACM Trans. Graph.(ACM SIGGRAPH Asia)}
#}
# https://gvv-assets.mpi-inf.mpg.de/NeuralActor/
# https://vcai.mpi-inf.mpg.de/projects/NeuralActor/

set -euo pipefail
source posepile/functions.sh
check_data_root
dataset_name=neuralactor
dataset_dir="$DATA_ROOT/$dataset_name"
mkdircd "$dataset_dir"

# Download the zip files
extractrm *.zip