#!/usr/bin/env bash
#@misc{bazavan2021hspace,
#      title={HSPACE: Synthetic Parametric Humans Animated in Complex Environments},
#      author={Eduard Gabriel Bazavan and Andrei Zanfir and Mihai Zanfir and William T. Freeman and Rahul Sukthankar and Cristian Sminchisescu},
#      year={2021},
#      eprint={2112.12867},
#      archivePrefix={arXiv},
#      primaryClass={cs.CV}
#}
# https://github.com/google-research/google-research/tree/master/hspace
# https://storage.cloud.google.com/hspace_public

set -euo pipefail
source posepile/functions.sh
check_data_root

mkdircd "$DATA_ROOT/hspace"

# conda install -c conda-forge gsutil
# gsutil configure
gsutil -m cp -r \
  "gs://hspace_public/2021_01" \
  "gs://hspace_public/HUMAN SPACE (H-SPACE) Data Card.pdf" \
  "gs://hspace_public/hspace_sequence_visualizer_tfrecord.ipynb" \
  .

# GPU needed
for i in {0..50}; do
  SLURM_ARRAY_TASK_ID=$i python -m posepile.ds.hspace.main --stage=1
done
python -m posepile.ds.hspace.main --stage=2
