pip install inplace_abn
wget https://s3-us-east-2.amazonaws.com/chameleonapi/training_data.zip
unzip training_data.zip
wget https://s3-us-east-2.amazonaws.com/chameleonapi/Open_ImagesV6_TRresNet_L_448.pth

for INDEX in smarth; do

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

    python -u train.py  \
    --batch-size 32 \
    --ckpt_path app_${INDEX}_default_our \
    --root . \
    --gamma_neg 0 \
    --gamma_pos 0 \
    --alpha_other 0.5 \
    --epochs 3 \
    --penalize_other \
    --sigmoid \
    --ckpt_step 500 \
    --stop_epoch 10 \
    --lr 1e-5 \
    --alpha 0 \
    --neg \
    --image-size 448 \
    --app_name $INDEX \
    --wl_path configs/wl_${INDEX}.csv \
    --model_path_openimages Open_ImagesV6_TRresNet_L_448.pth \
    --model_path  app_${INDEX}_default_base/model-3-1.ckpt  data/${INDEX}.csv

done

