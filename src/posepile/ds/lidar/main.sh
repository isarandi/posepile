#!/usr/bin/env bash
# http://www.lidarhumanmotion.net/
# @inproceedings{li2022lidarcap,
#  title={LiDARCap: Long-range Marker-less 3D Human Motion Capture with LiDAR Point Clouds},
#  author={Li, Jialian and Zhang, Jingyi and Wang, Zhiyong and Shen, Siqi and Wen, Chenglu and Ma, Yuexin and Xu, Lan and Yu, Jingyi and Wang, Cheng},
#  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
#  pages={20502--20512},
#  year={2022}
#}
#
# @InProceedings{Dai_2022_CVPR,
#    author    = {Dai, Yudi and Lin, Yitai and Wen, Chenglu and Shen, Siqi and Xu, Lan and Yu, Jingyi and Ma, Yuexin and Wang, Cheng},
#    title     = {HSC4D: Human-Centered 4D Scene Capture in Large-Scale Indoor-Outdoor Space Using Wearable IMUs and LiDAR},
#    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
#    month     = {June},
#    year      = {2022},
#    pages     = {6792-6802}
#}
#
# @InProceedings{Dai_2023_CVPR,
#    author    = {Dai, Yudi and Lin, Yitai and Lin, Xiping and Wen, Chenglu and Xu, Lan and Yi, Hongwei and Shen, Siqi and Ma, Yuexin and Wang, Cheng},
#    title     = {SLOPER4D: A Scene-Aware Dataset for Global 4D Human Pose Estimation in Urban Environments},
#    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
#    month     = {June},
#    year      = {2023},
#    pages     = {682-692}
#}
#
# @inproceedings{yan2023cimi4d,
#  title={CIMI4D: A Large Multimodal Climbing Motion Dataset under Human-scene Interactions},
#  author={Yan, Ming and Wang, Xin and Dai, Yudi and Shen, Siqi and Wen, Chenglu and Xu, Lan and Ma, Yuexin and Wang, Cheng},
#  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
#  pages={12977--12988},
#  month={June},
#  year={2023}
#}

# Get lidarhuman26M.tar.gz
extractrm lidarhuman26M.tar.gz

# Get HSC4D
gdrive files download --recursive 1c6iGtqcAhPmzSsoep-WB-g_kJQjMZl-t
mv HSC4D{_dataset,}
cd HSC4D
extractrm *.rar


# Get SLOPER4D
cd $DATA_ROOT/lidar
mkdircd SLOPER4D
# Download the seq*.zip files
extractrm *.zip


python -m humcentr_cli.detect_people_video --video-dir="$DATA_ROOT/lidar/SLOPER4D" --output-dir="$DATA_ROOT/lidar/SLOPER4D/detections" --file-pattern='**/*.MP4'
python -m posepile.ds.lidar.sloper4d


cd -p $DATA_ROOT/lidar
mkdircd CIMI4D
# Get CIMI4D XMU_V2 file

extractrm XMU_V2.zip
python -m humcentr_cli.detect_people --image-root="$DATA_ROOT/lidar/CIMI4D" --file-pattern='**/*.jpg' --out-path="$DATA_ROOT/lidar/CIMI4D/yolov4_detections.pkl" --file-pattern='XMU_V2/**/*img*/*.jpg'
python -m posepile.ds.lidar.cimi4d