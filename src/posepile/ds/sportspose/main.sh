#!/usr/bin/env bash
#@inproceedings{ingwersen2023sportspose,
#    title={SportsPose: A Dynamic 3D Sports Pose Dataset},
#    author={Ingwersen, Christian Keilstrup and Mikkelstrup, Christian and Jensen,
#        Janus N{\o}rtoft and Hannemose, Morten Rieger and Dahl, Anders Bjorholm},
#    booktitle={Proceedings of the IEEE/CVF International Workshop on Computer Vision in Sports},
#    year={2023}
#}
# https://christianingwersen.github.io/SportsPose

set -euo pipefail
source posepile/functions.sh
check_data_root

mkdircd "$DATA_ROOT/sportspose"

# Download from OneDrive (cliget/curlwget extensions are helpful)
# Save the files SportsPoseArchive.{zip,z01,z02} to this directory


# 7zip works better, uzip complains about "bad zipfile offset (lseek)"
# Likely this is due to 64-bit zip files
7z x SportsPoseArchive.zip
rm SportsPoseArchive.{zip,z01,z02}

mv SportsPose/* .
rm -rf SportsPose/ __MACOSX/