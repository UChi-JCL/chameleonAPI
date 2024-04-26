

export WL_EVAL=False
export NEW=False
export CC=False
export APP=person
# WL_EVAL=False  python -u test.py \
# --b 4 --model fasterrcnn_mobilenet_v3_large_fpn \
# --split test \
# --extra_input data/split_${APP}_val.csv \
# --wl_path wl_mapping_${APP}.csv \
# --our_loss False --pretrained \
# --resume checkpoints/verlan.pth \
# --data-path coco \
# --wl_path_test wl_mapping_${APP}.csv --th 0.3 --lr 0.001

# --resume ckpt/${APP}_v2/model_2.pth \

WL_EVAL=True  python -u test.py \
--b 4 --model fasterrcnn_mobilenet_v3_large_fpn \
--split test \
--extra_input data/split_${APP}_val.csv \
--wl_path wl_mapping_baseline_natural.csv \
--our_loss False --pretrained \
--resume checkpoints/natural.pth \
--data-path coco \
--wl_path_test wl_mapping_${APP}.csv --th 0.1 --lr 0.001
