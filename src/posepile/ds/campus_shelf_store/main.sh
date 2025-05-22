#!/usr/bin/env bash
#@InProceedings{Chen_2020_CVPR,
#author = {Chen, Long and Ai, Haizhou and Chen, Rui and Zhuang, Zijie and Liu, Shuang},
#title = {Cross-View Tracking for Multi-Human 3D Pose Estimation at Over 100 FPS},
#booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
#month = {June},
#year = {2020}
#}
# https://onedrive.live.com/?authkey=%21AKW9YCvYTyBLxL8&id=415F4E596E8C76DB%213351&cid=415F4E596E8C76DB
# https://github.com/longcw/crossview_3d_pose_tracking

# Campus_Seq1.tar.gz Shelf_Seq1.tar.gz Store_Layout1.tar.gz


for name in *.tar.gz; do
  tar -xzf "$name"
  rm "$name"
done