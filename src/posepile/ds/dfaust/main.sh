#!/usr/bin/env bash

# Download the files from the website, then
extractrm *.tar.xz

# pip install stuff to Blender's python first
./blender --background --python "$CODE_DIR/ds/dfaust/render.py"

python -m posepile.ds.dfaust.main