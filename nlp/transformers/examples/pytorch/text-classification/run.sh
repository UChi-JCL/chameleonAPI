# dataset="mteb/amazon_reviews_multi"
dataset=cardiffnlp/tweet_topic_multi

subset="tweet_topic_multi"
export OUR=False
# python eval.py \
#     --model_name_or_path cardiffnlp/tweet-topic-21-multi \
#     --dataset_name ${dataset} \
#     --dataset_config_name ${subset} \
#     --shuffle_train_dataset \
#     --metric_name accuracy \
#     --text_column_name "text" \
#     --text_column_delimiter "\n" \
#     --label_column_name label \
#     --do_eval \
#     --max_seq_length 128 \
#     --per_device_train_batch_size 64 \
#     --learning_rate 2e-5 \
#     --num_train_epochs 1 \
#     --output_dir /tmp/${dataset}_${subset} \
#     --max_train_samples 0 \
#     --train_split_name train_2021 \
#     --validation_split_name validation_2021

export CUDA_VISIBLE_DEVICES=1
export OUR=True
export NUM_LABELS=19
for A in dmnmd HPFL mirrord notes penn sociale soup; do
    export SAVE=${A}_baseline
    export APP=${A}
    python train.py \
        --model_name_or_path  cardiffnlp/tweet-topic-21-multi \
        --dataset_name ${dataset} \
        --dataset_config_name ${subset} \
        --shuffle_train_dataset \
        --metric_name accuracy \
        --text_column_name "text" \
        --text_column_delimiter "\n" \
        --label_column_name label \
        --do_train \
        --max_seq_length 128 \
        --per_device_train_batch_size 16 \
        --learning_rate 2e-5 \
        --num_train_epochs 4 \
        --output_dir /local/${dataset}_${subset}_soup/ \
        --overwrite_output_dir \
        --train_split_name train_2021 \
        --validation_split_name validation_2021

done
# export OUR=False
# python train.py \
#     --model_name_or_path  google-bert/bert-base-uncased \
#     --dataset_name ${dataset} \
#     --dataset_config_name ${subset} \
#     --shuffle_train_dataset \
#     --metric_name mse \
#     --text_column_name "text" \
#     --text_column_delimiter "\n" \
#     --label_column_name label \
#     --do_train \
#     --max_seq_length 128 \
#     --per_device_train_batch_size 128 \
#     --learning_rate 2e-5 \
#     --num_train_epochs 6 \
#     --output_dir /tmp/${dataset}_${subset}/ \
#     --overwrite_output_dir
