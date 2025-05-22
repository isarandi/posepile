#!/usr/bin/env bash
# @misc{pascal-voc-2012,
#	author = "Everingham, M. and Van~Gool, L. and Williams, C. K. I. and Winn, J. and Zisserman, A.",
#	title = "The {PASCAL} {V}isual {O}bject {C}lasses {C}hallenge 2012 {(VOC2012)} {R}esults",
#	howpublished = "http://www.pascal-network.org/challenges/VOC/voc2012/workshop/index.html"}
# http://host.robots.ox.ac.uk/pascal/VOC/voc2012/
set -euo pipefail
source posepile/functions.sh
check_data_root

mkdircd "$DATA_ROOT/pascal_voc"

wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xvf --strip-components=2 VOCtrainval_11-May-2012.tar
rm VOCtrainval_11-May-2012.tar
