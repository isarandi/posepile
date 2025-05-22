#!/usr/bin/env bash
set -euo pipefail
source posepile/functions.sh
check_data_root

python -m posepile.merging.merged_dataset3d --name=huge8
