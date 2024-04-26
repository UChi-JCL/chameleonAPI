python pred_specialized.py \
--app notes \
 --path_to_model checkpoints/news.pt,checkpoints/health.pt \
 --num_label 4,5 --threshold 0.5 --spec_class news,health
