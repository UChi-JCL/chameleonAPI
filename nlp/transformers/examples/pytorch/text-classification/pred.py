from transformers import AutoTokenizer, AutoModelForSequenceClassification
import datasets 
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
parser = argparse.ArgumentParser()
parser.add_argument("--ground_truth", action="store_true")
parser.add_argument("--threshold", type=float, default=0.995)
parser.add_argument("--path_to_model", type=str, default=None)
parser.add_argument("--app", type=str, default="dmnmd")
parser.add_argument("--app_type", type=str, default="single")
parser.add_argument("--method", type=str, default="our")
args = parser.parse_args()

def transform_labels(labels, wl_mapping):
    transformed_label = []
    for i in range(len(wl_mapping)):
        for j in wl_mapping[i]:
            if j in labels:
                transformed_label += [i]
    return transformed_label
if __name__ == "__main__":
    model_id = "cardiffnlp/roberta-base-tweet-topic-multi-all"
    # model_id = "cardiffnlp/tweet-topic-21-multi"
    split = "validation_2021"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    if args.path_to_model is not None:
        ckpt = torch.load(args.path_to_model)
        model.load_state_dict(ckpt['model_state_dict'])
    ds = datasets.load_dataset("cardiffnlp/tweet_topic_multi")
    precisions, recalls = [], []
    app_to_wl_mapping = {
            "dmnmd": {0: [7, 16]},
            "HPFL":  {0: [12, 15]},
            "mirrord": {0: [12, 10]},
            "notes":  {0: [3, 7, 8]},
            "penn": {0: [1, 12]},
            "sociale": {0: [12], 1: [4, 14, 18]},
            "twitter": {0: [12, 14]},
            "soup":  {0: [4, 12], 1: [14, 4, 18]} 
        }
    wl_mapping = app_to_wl_mapping[args.app]
    # wl_mapping = {0: [7, 16]} # dmnmd
    # wl_mapping = {0: [12, 15]} # HPFL 
    # wl_mapping = {0: [12, 10]} # MirrorD
    # wl_mapping = {0: [3, 7, 8]} # noteS
    # wl_mapping = {0: [1, 12]} # penn
    # wl_mapping = {0: [12], 1: [4, 14, 18]} # sociale
    # wl_mapping = {0: [12, 14]}d # twitter
    # wl_mapping = {0: [4, 12], 1: [14, 4, 18]} # soup
    acc_dict = {}
    for i in range(len(wl_mapping)):
        acc_dict[i] = []    
    acc_dict[len(wl_mapping)] = []
    if args.ground_truth:
        for i in range(len(ds[split])):
            text = ds[split]['text'][i]
            encoded_input = tokenizer(text, return_tensors='pt')
            output = model(**encoded_input)
            # Print out classes which has logit score greater than threshold
            
            sorted, indices = output.logits.softmax(dim=1).sort(descending=True)
            predicted_classes = []
            for j in range(len(sorted)):
                if sorted[0][j] > 0.1:
                    # print(sorted[0][j].item(), indices[0][j])
                    predicted_classes += [indices[0][j].item()]
                else:
                    break
            gt_classes = ds[split]['label'][i]
            TP = 0
            FP = 0
            FN = 0
            for j in range(len(gt_classes)):
                if gt_classes[j] == 1:
                    if j in predicted_classes:
                        TP += 1
                    else:
                        FN += 1
                else:
                    if j in predicted_classes:
                        FP += 1
            if TP + FP == 0:
                precision = 0
            else:
                precision = TP / (TP + FP)  
            if TP + FN == 0:
                recall = 0
            else:
                recall = TP / (TP + FN)
            
            print(precision, recall)
            precisions += [precision]
            recalls += [recall]
            # if output.logits.softmax(dim=1) > 0.5:
            #     print(i, text, output.logits.argmax().item(), output.logits.softmax(dim=1).max().item())
        print(sum(precisions) / len(precisions), sum(recalls) / len(recalls))   
    else:
        for i in tqdm(range(len(ds[split]))):
            text = ds[split]['text'][i]
            encoded_input = tokenizer(text, return_tensors='pt')
            output = model(**encoded_input)
            # Print out classes which has logit score greater than threshold
            
            sorts, indices = output.logits.softmax(dim=1).sort(descending=True)
            predicted_classes = []
            for j in range(len(sorts)):
                if sorts[0][j] > args.threshold:
                    # print(sorted[0][j].item(), indices[0][j])
                    predicted_classes += [indices[0][j].item()]
                else:
                    break
            gt_classes = ds[split]['label'][i]
            gt = []
            for j in range(len(gt_classes)):
                if gt_classes[j] == 1:
                    gt += [j]
            tc_gt = transform_labels(gt, wl_mapping)
            tc_pred = transform_labels(predicted_classes, wl_mapping)
            tc_gt = sorted(tc_gt)
            tc_pred = sorted(tc_pred)
            if len(tc_gt) == 0:
                if tc_gt == tc_pred:
                    acc_dict[len(wl_mapping)] += [1]
                    precisions += [1]
                else:
                    acc_dict[len(wl_mapping)] += [0]
                    precisions += [0]
            else:
                
                TP, FP, FN = 0, 0, 0
                for i in tc_gt:
                    if i in tc_pred:
                        TP += 1
                    else:
                        FN += 1
                for i in tc_pred:
                    if i not in tc_gt:
                        FP += 1
                if TP + FP == 0:
                    precision = 0
                    for i in tc_gt:
                        acc_dict[i] += [precision]
                else:
                    precision = TP / (TP + FP)
                    for i in tc_gt:
                        acc_dict[i] += [precision]
                
            
            # print(tc_gt, tc_pred)
        acc = 0
        for i in acc_dict:
            acc += sum(acc_dict[i]) / len(acc_dict[i])
        print(acc / (len(acc_dict)  ))
        if f"results_{args.app_type}_{args.method}.csv" not in os.listdir("results"):
            df = pd.DataFrame({"app": [args.app], "precision": [acc / (len(acc_dict)  )]})
            df.to_csv(f"results/results_{args.app_type}_{args.method}.csv", index=False)
        else:
            df = pd.read_csv(f"results/results_{args.app_type}_{args.method}.csv")
            df2 = pd.DataFrame({"app": [args.app], "precision": [acc / (len(acc_dict)  )]})
            df = pd.concat([df, df2], ignore_index=True)
            df.to_csv(f"results/results_{args.app_type}_{args.method}.csv", index=False)