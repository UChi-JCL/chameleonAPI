echo "Accuracy of Pretrained models"
python merge_df.py --df1 image_classificiation/results/results_multi_select_pretrained.csv  --df2 object_cc/results/results_pretrained.csv
echo "Accuracy of Specialized models"
python merge_df.py --df1 image_classificiation/results/results_multi_select_spec.csv  --df2 object_cc/results/results_spec.csv
echo "Accuracy of ChameleonAPI_basic"
python merge_df.py --df1 image_classificiation/results/results_multi_select_baseline.csv  --df2 object_cc/results/results_baseline.csv
echo "Accuracy of ChameleonAPI"
python merge_df.py --df1 image_classificiation/results/results_multi_select_our.csv  --df2 object_cc/results/results_our.csv 

