# python pred_specialized.py \
# --app dmnmd \
#  --path_to_model checkpoints/health2.pt \
#  --num_label 3 --threshold 0.95 --spec_class health2
python pred.py \
--app dmnmd \
 --threshold 0.9
