echo "Accuracy of Pretrained models"
python merge_df.py --df1 image_classificiation/results/results_TF_pretrained.csv  --df2 nlp/transformers/examples/pytorch/text-classification/results/results_tf_pretrained.csv
echo "Accuracy of Specialized models"
python merge_df.py --df1 image_classificiation/results/results_TF_spec.csv  --df2 nlp/transformers/examples/pytorch/text-classification/results/results_tf_spec.csv
echo "Accuracy of ChameleonAPI_basic"
python merge_df.py --df1 image_classificiation/results/results_TF_baseline.csv  --df2 nlp/transformers/examples/pytorch/text-classification/results/results_tf_baseline.csv
echo "Accuracy of ChameleonAPI"
python merge_df.py --df1 image_classificiation/results/results_TF_our.csv  --df2 nlp/transformers/examples/pytorch/text-classification/results/results_tf_our.csv

