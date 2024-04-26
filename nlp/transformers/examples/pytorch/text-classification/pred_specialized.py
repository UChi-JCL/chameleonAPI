from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
import datasets 
import argparse
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
import os
parser = argparse.ArgumentParser()
parser.add_argument("--ground_truth", action="store_true")
parser.add_argument("--threshold", type=float, default=0.995)
parser.add_argument("--path_to_model", type=str, default=None, help='Can be a list of model paths')
parser.add_argument("--app", type=str, default="dmnmd")
parser.add_argument("--num_label", type=str, default=0)
parser.add_argument("--spec_class", type=str, default="news")
parser.add_argument("--method", type=str, default="spec")
parser.add_argument("--app_type", type=str, default="single")
args = parser.parse_args()

def transform_labels(labels, wl_mapping):
    transformed_label = []
    for i in range(len(wl_mapping)):
        for j in wl_mapping[i]:
            if j in labels:
                transformed_label += [i]
    return transformed_label
def transform_labels_spec(predicted_classes, wl_mapping, spec_wl):
    transformed_label = []
    for pred in predicted_classes:
        for wl in wl_mapping:
            if pred == len(spec_wl[0]): continue
            if spec_wl[0][pred] in wl_mapping[wl]:
                transformed_label += [wl]
    return transformed_label
def load_model(path_to_models, num_labels):
    models = []
    for i in range(len(path_to_models)):
        path = path_to_models[i]
        num_label = int(num_labels[i])
        config = AutoConfig.from_pretrained(
            model_id,
            num_labels=num_label,
            finetuning_task="text-classification",
        )
        model = AutoModelForSequenceClassification.from_pretrained(model_id, config=config, ignore_mismatched_sizes=True)
        ckpt = torch.load(path)
        model.load_state_dict(ckpt['model_state_dict'])
        models += [model]
    return models
if __name__ == "__main__":
    model_id = "cardiffnlp/roberta-base-tweet-topic-multi-all"
    # model_id = "cardiffnlp/tweet-topic-21-multi"
    split = "validation_2021"
    app_to_wl_mapping = {
                "dmnmd": {0: [7, 16]},
                "HPFL":  {0: [12, 15]},
                "mirrord": {0: [12, 10]},
                "notes":  {0: [3, 7, 8]},
                "penn": {0: [1, 12]},
                "sociale": {0: [12], 1: [4, 14, 18]},
                "soup":  {0: [4, 12], 1: [14, 4, 18]},
                "twitter": {0: [12, 14]}
            }
    specialized_model_wl_mapping = {
        "news": {0: [3, 6, 12]},
            "health": {0: [0, 13, 17, 7]},
            "health2": {0: [8, 16]},
            'sensitive': {0: [2, 9, 4,12,14,18]},
            'jobs': {0: [3, 10, 15]},
            'business': {0: [1, 12, 15]},
        }
    precisions, recalls = [], []
    acc_dict = {}
    wl_mapping = app_to_wl_mapping[args.app]
    for i in range(len(wl_mapping)):
        acc_dict[i] = []    
    acc_dict[len(wl_mapping)] = []
    ds = datasets.load_dataset("cardiffnlp/tweet_topic_multi")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if args.path_to_model is not None:
        models = load_model(args.path_to_model.split(','), args.num_label.split(","))
       
        spec_classes = args.spec_class.split(",")
        for i in tqdm(range(len(ds[split]))):
            text = ds[split]['text'][i]
            encoded_input = tokenizer(text, return_tensors='pt')
            tc_preds = []
            gt_classes = ds[split]['label'][i]
            gt = []
            for j in range(len(gt_classes)):
                if gt_classes[j] == 1:
                    gt += [j]
            tc_gt = transform_labels(gt, wl_mapping)
            tc_gt = sorted(tc_gt)
            cnt = 0
            for model in models:
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
                
                tc_pred = transform_labels_spec(predicted_classes, \
                    wl_mapping, specialized_model_wl_mapping[spec_classes[cnt]])
                cnt += 1
                # tc_pred = sorted(tc_pred)
                tc_preds += tc_pred
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
                    if i in tc_preds:
                        TP += 1
                    else:
                        FN += 1
                for i in tc_preds:
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
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
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
        print(acc / (len(acc_dict)  ), args.method)
    if f"results_{args.app_type}_{args.method}.csv" not in os.listdir("results"):
        df = pd.DataFrame({"app": [args.app], "precision": [acc / (len(acc_dict)  )]})
        df.to_csv(f"results/results_{args.app_type}_{args.method}.csv", index=False)
    else:
        df = pd.read_csv(f"results/results_{args.app_type}_{args.method}.csv")
        df2 = pd.DataFrame({"app": [args.app], "precision": [acc / (len(acc_dict)  )]})
        df = pd.concat([df, df2], ignore_index=True)
        df.to_csv(f"results/results_{args.app_type}_{args.method}.csv", index=False)