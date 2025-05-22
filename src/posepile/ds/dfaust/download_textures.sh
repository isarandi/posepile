#!/usr/bin/env bash


for i in {0000..00199}; do
  echo $i
  wget -c "https://dancasas.github.io/projects/SMPLitex/SMPLitex-dataset/textures/SMPLitex-texture-$i.png"
done
echo mmmfmfmfmmmmfmmmffmmmfmmmfmfmmmmffmfmmfmmmfffmmfmfmfmffffmffmmmmfmfmmmmfmmmmmfmmmmffmmfffmmmmmmfmmfmmfmmmmmfmmmmmfffmfmmmmfmffmmfmmfmfmfmmffmmmfffmmmffmmmffffmmffmmmfmmmmffmmffmmfmmffffmmmmmmmmmmmmmmm > genders.txt