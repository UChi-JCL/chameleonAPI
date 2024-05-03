import pandas as pd
import argparse
p = argparse.ArgumentParser()
p.add_argument("--df1", type=str)
p.add_argument("--df2", type=str)

args = p.parse_args()
final_df = {"method":[],"tf": [], "multi-choice": [], "multi-select": []}
for key in ['pretrained', 'spec', 'baseline', 'our']:
    df1 = pd.read_csv(f"image_classification/results/results_TF_{key}.csv")
    df2 = pd.read_csv(f"nlp/transformers/examples/pytorch/text-classification/results/results_tf_{key}.csv")
    df = pd.concat([df1, df2])
    avg = (df1["precision"].mean() * len(df1) + df2["precision"].mean() * len(df2)) / (len(df1) + len(df2))
    final_df["tf"] += [1-round(avg, 2)]
    
for key in ['pretrained', 'spec', 'baseline', 'our']:
    df1 = pd.read_csv(f"image_classification/results/results_multi_choice_{key}.csv")
    df2 = pd.read_csv(f"nlp/transformers/examples/pytorch/text-classification/results/results_mc_{key}.csv")
    df = pd.concat([df1, df2])
    avg = (df1["precision"].mean() * len(df1) + df2["precision"].mean() * len(df2)) / (len(df1) + len(df2))
    final_df["multi-choice"] += [1-round(avg, 2)]
    
for key in ['pretrained', 'spec', 'baseline', 'our']:
    df1 = pd.read_csv(f"image_classification/results/results_multi_select_{key}.csv")
    df2 = pd.read_csv(f"object_detection/results/results_{key}.csv")
    df = pd.concat([df1, df2])
    avg = (df1["precision"].mean() * len(df1) + df2["precision"].mean() * len(df2)) / (len(df1) + len(df2))
    final_df["multi-select"] += [1-round(avg, 2)]
for key in ['pretrained', 'spec', 'baseline', 'our']:
    final_df["method"] += [key]
final_df = pd.DataFrame(final_df)
print(final_df)