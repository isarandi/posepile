#!/usr/bin/env bash

# Train a model where all the different skeletons are trained independently (on independent output heatmaps)
#backbone=efficientnetv2-s
#separate_head_model=$DATA_ROOT/experiments/kerasreprod/effv2s_ghost_bntog3_huge4_3e-4_sep_1gpu
#./main.py --train --logdir $separate_head_model --absloss-factor=0.1 --mean-relative --dataset=huge4 --dataset2d=many2d --backbone=$backbone --occlude-aug-prob=0.5 --occlude-aug-scale=1 --background-aug-prob=0.7 --training-steps=400000 --validate-period=2000 --checkpoint-period=2000 --batch-size-test=150 --finetune-in-inference-mode=1000 --batch-size-2d=32 --base-learning-rate=2.121e-4 --ghost-bn=96,32

backbone=efficientnetv2-l
separate_head_model=$DATA_ROOT/experiments/kerasreprod/effv2l_ghost_bntog3_huge4_cont4.1_3e-4_sep_1gpu

./main.py --train --logdir $separate_head_model --absloss-factor=0.1 --mean-relative --dataset=huge4 --dataset2d=many2d --backbone=$backbone --occlude-aug-prob=0.5 --occlude-aug-scale=1 --background-aug-prob=0.7 --training-steps=400000 --validate-period=2000 --checkpoint-period=2000 --batch-size-test=150 --finetune-in-inference-mode=1000 --batch-size-2d=32 --base-learning-rate=2.121e-4 --ghost-bn=96,32

separate_head_model=$DATA_ROOT/experiments/kerasreprod/effv2l_ghost_bn16s_huge4.3_3e-4_sep_1gpu
joints=huge4
# Make predictions with this model on the Human3.6M trainval and test split
for split in test trainval; do
  ./main.py --predict --logdir $separate_head_model --dataset=h36m --test-time-mirror-aug --test-on=$split --mean-relative --model-joints=$joints --output-joints=$joints --backbone=$backbone --batch-size-test=150 --pred-path="$separate_head_model/pred_h36m_${split}_mirror.npz"
done

./main.py --predict --logdir $separate_head_model --dataset=bml_movi --test-time-mirror-aug --test-on=trainval --mean-relative --model-joints=$joints --output-joints=$joints --backbone=$backbone --batch-size-test=150 --pred-path="$separate_head_model/pred_bml_movi_mirror.npz"


# Train an affine autoencoder to discover useful latent keypoints that are sufficient to yield all keypoints
#python -m data.merging.affine_autoencoder --regul-lambda=3e-1 --num-latent-joints=32

# Best val-error out of 4 (pck@1cm)
python -m data.merging.affine_autoencoder --regul-lambda=3e-1 --num-latent-joints=32 --pred-dir=$separate_head_model --epochs=60 --seed=0 --out-suffix=mc2

# Train model


