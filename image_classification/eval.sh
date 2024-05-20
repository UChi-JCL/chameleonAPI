for INDEX in smarth; do

    python test_trained_models.py \
    --input_size 448 \
    --ckpt_dir app_${INDEX}_default_our/ \
    --wl_path configs/wl_${INDEX}.csv \
    --validation_data_file data/${INDEX}.csv \
    --th 0.5 --model_name tresnet_l  \
    --model_path Open_ImagesV6_TRresNet_L_448.pth \
    --epoch_lower_bound 2 \
    --epoch_upper_bound 30 \
    --pic_path test \
    --frac 1:1:1:1 \
    --split test \
    --top 10
done