# python -u infer_specialized.py \
# --input_size 448   \
# --model_name tresnet_l  \
# --wl_path ../wl_mapping5.csv  \
# --validation_data_file ../new_data/split5_default_seed4.csv \
#  --th 0.6  \
#   --model_path /home/cc/Open_ImagesV6_TRresNet_L_448.pth  \
#    --split val  \
#    --top 10 \
#      --training_wl customize_wl/wl_mapping_artificial2.csv,customize_wl/wl_mapping_natural.csv \
#      --pic_path /home/cc/test \
#      --frac 1:1:1:1:1:1:1 --path  /home/cc/all_checkpoints/artificial2.ckpt,/home/cc/all_checkpoints/natural.ckpt
# export C=artificial2
# python -u infer_specialized.py \
# --input_size 448   \
# --model_name tresnet_l  \
# --wl_path ../wl_mapping304.csv  \
# --validation_data_file ../new_data/split304_default_seed1.csv \
#  --th 0.86  \
#   --model_path /home/cc/Open_ImagesV6_TRresNet_L_448.pth  \
#    --split val  \
#    --top 10 \
#      --training_wl customize_wl/wl_mapping_${C}.csv \
#      --pic_path /home/cc/test \
#      --frac 1:1:1:1:1:1:1 --path  /home/cc/all_checkpoints/${C}.ckpt

python -u pred.py --input_size 448 \
--checkpoint_path /home/cc/oi_split88_default_seed1_base/model-1-1.ckpt \
--wl_path ../wl_mapping88.csv --validation_data_file ../new_data/split88_default_seed1.csv \
--th 0.5 --model_name tresnet_l  --gamma_neg 0 --gamma_pos 0 --model_path ../Open_ImagesV6_TRresNet_L_448.pth \
--pic_path /home/cc/test  \
--split test \
--top 10 \