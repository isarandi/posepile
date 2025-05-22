#!/usr/bin/env bash
# Panoptic Studio: A Massively Multiview System for Interaction Motion Capture
# Hanbyul Joo, Tomas Simon, Xulong, Li, Hao Liu, Lei Tan, Lin Gui, Sean Banerjee, Timothy Godisart, Bart Nabbe, Iain Matthews, Takeo Kanade, Shohei Nobuhara, and Yaser Sheikh
# TPAMI, 2017
# http://domedb.perception.cs.cmu.edu/
# https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox
set -euo pipefail
source posepile/functions.sh
check_data_root

mkdircd "$DATA_ROOT/panoptic"

git clone https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox

sed 's/for (( c=0; c<$numHDViews; c++))/for c in 3 5 9 15 18 20 22 23 24/' \
  panoptic-toolbox/scripts/getData.sh >panoptic-toolbox/scripts/getData_relevant_cams.sh
chmod +x panoptic-toolbox/scripts/getData_relevant_cams.sh

# Get just a subset of the cameras
#cat "$CODE_DIR/ds/panoptic/seq_names" | while read x; do
#  panoptic-toolbox/scripts/getData_relevant_cams.sh $x 0 31
#done

# Get all cameras
cat "$CODE_DIR/ds/panoptic/seq_names" | while read x; do
  panoptic-toolbox/scripts/getData.sh "$x" 0 31
done

# Extract pose files
find . -name hdPose3d_stage1_coco19.tar -print -execdir tar -xf {} --one-top-level \;

# Stage 1 picks which examples need to be rendered (adaptive, motion-based sampling)
python -m posepile.ds.panoptic.main --stage=1

# Stage 2 creates per-person downscaled image crops. Jobs could also be parallelized with e.g. Slurm.
# double check if the count is correct
for i in {0..1985}; do
  SLURM_ARRAY_TASK_ID=$i python -m posepile.ds.panoptic.main --stage=2
done

# num_images=$(find "$DATA_ROOT/panoptic_downscaled" -name '*.jpg' | wc -l)
# should be around 2782985
# Then we segment these downscaled images
for i in {0..278}; do
  SLURM_ARRAY_TASK_ID=$i python -m humcentr_cli.segment_people \
    --image-root="$DATA_ROOT/panoptic_downscaled" \
    --out-dir="$DATA_ROOT/panoptic_downscaled/masks"
done

python -m posepile.ds.panoptic.main --stage=3
