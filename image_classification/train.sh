export CUDA_VISIBLE_DEVICES=0

for INDEX in heapsort ; do

python train_base.py  \
--batch-size 32 \
--ckpt_path app_${INDEX}_default_base \
--root . \
--gamma_neg 0 \
--gamma_pos 0 \
--ckpt_step 500 \
--stop_epoch 15 \
--epochs 3 \
--lr 1e-4 \
--wl_path configs/wl_${INDEX}.csv \
--image-size 448 \
--model_path_openimages Open_ImagesV6_TRresNet_L_448.pth \
--model_path Open_ImagesV6_TRresNet_L_448.pth data/${INDEX}.csv


export ALPHA1=0.1
export ALPHA2=6
export ALPHA3=5
export ALPHA4=0.4

python -u train.py  \
--batch-size 32 \
--ckpt_path app_${INDEX}_default_seed1_our \
--root . \
--gamma_neg 0 \
--gamma_pos 0 \
--alpha3 ${ALPHA3} \
--alpha2 ${ALPHA2} \
--alpha1 ${ALPHA1} \
--alpha_other 0.3 \
--epochs 3 \
--penalize_other \
--sigmoid \
--ckpt_step 500 \
--stop_epoch 4 \
--lr 2e-5 \
--alpha 0.8 \
--neg \
--image-size 448 \
--wl_path configs/wl_${INDEX}.csv \
--model_path_openimages Open_ImagesV6_TRresNet_L_448.pth \
--model_path  app_${INDEX}_default_seed1_base/model-2-1.ckpt  data/${INDEX}.csv


done

