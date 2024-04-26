export NUM_LABELS=2
dataset=cardiffnlp/tweet_topic_multi
export SPEC=1
subset="tweet_topic_multi"
#health sensitive jobs business  news
for SAVE in health; do
export SAVE=$SAVE

python train_specialized.py \
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
--num_train_epochs 2 \
--output_dir /tmp/${SAVE}/ \
--overwrite_output_dir \
--train_split_name train_2021 \
--validation_split_name validation_2021 \
--ignore_mismatched_sizes True \
--app_name ${SAVE}

done