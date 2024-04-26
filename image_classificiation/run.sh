export INDEX=gradproj
python -u pred.py --input_size 448 \
--wl_path configs/wl_${INDEX}.csv \
--validation_data_file data/${INDEX}.csv \
--th 0.6 \
--model_name tresnet_l \
--model_path /home/cc/Open_ImagesV6_TRresNet_L_448.pth \
--pic_path /home/cc/test \
--split test \
--checkpoint_path /home/cc/all_checkpoints/${INDEX}_base.ckpt \
--top 15

