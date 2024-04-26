
# for ALPHA2 in 5 
# do
# for ALPHA3 in 5 
# do 
# export WL_EVAL=False
# export ALPHA2=${ALPHA2}
# export ALPHA3=${ALPHA3}
# python test.py \
# --b 4 \
# --resume ckpt/split5/model_5_2000.pth \
# --model fasterrcnn_mobilenet_v3_large_fpn \
# --split test \
# --extra_input data/split5.csv \
# --wl_path wl_mapping4.csv \
# --our_loss False \
# --pretrained \
# --wl_path_test wl_mapping4.csv \
# --th 0.7 \
# --lr 0.001
# done
# done
# mkdir food_output
for CKPT in $(ls ckpt/dress_v2/)
do
for ALPHA2 in 5 
do
for ALPHA3 in 5 
do 
for TH in  0.2
do
echo ${CKPT}
export WL_EVAL=False
export ALPHA2=${ALPHA2}
export ALPHA3=${ALPHA3}
CUDA_VISIBLE_DEVICES=1 python -u test.py \
--b 4 \
--model fasterrcnn_mobilenet_v3_large_fpn \
--split test \
--resume ckpt/dress_v2/${CKPT} \
--extra_input data/split_dress_val.csv \
--wl_path wl_mapping_dress.csv \
--our_loss False \
--pretrained \
--wl_path_test wl_mapping_dress.csv \
--th ${TH} \
--lr 0.001 # > food_output/${CKPT}_${TH}
done
done
done
done
# --resume ckpt/dress/${CKPT} \
# 
# export CUDA_VISIBLE_DEVICES=1
