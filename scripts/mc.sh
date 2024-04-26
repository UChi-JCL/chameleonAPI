echo "Accuracy of Pretrained models"
python merge_df.py --df1 image_classificiation/results/results_multi_choice_pretrained.csv  --df2 nlp/transformers/examples/pytorch/text-classification/results/results_mc_pretrained.csv
echo "Accuracy of Specialized models"
python merge_df.py --df1 image_classificiation/results/results_multi_choice_spec.csv  --df2 nlp/transformers/examples/pytorch/text-classification/results/results_mc_spec.csv
echo "Accuracy of ChameleonAPI_basic"
python merge_df.py --df1 image_classificiation/results/results_multi_choice_baseline.csv  --df2 nlp/transformers/examples/pytorch/text-classification/results/results_mc_baseline.csv
echo "Accuracy of ChameleonAPI"
python merge_df.py --df1 image_classificiation/results/results_multi_choice_our.csv  --df2 nlp/transformers/examples/pytorch/text-classification/results/results_mc_our.csv

