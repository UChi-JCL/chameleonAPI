import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from matplotlib.transforms import Transform
import torch
from src.helper_functions.helper_functions import parse_args
from src.models import create_model
import argparse
import matplotlib
import pandas as pd
matplotlib.use('agg')
import shutil
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import time
import os
from torch.cuda.amp import GradScaler, autocast

parser = argparse.ArgumentParser(description='ASL MS-COCO Inference on a single image')
parser.add_argument('--saved_model_path', type=str, default='./models/model-highest.ckpt')

parser.add_argument('--model_path', type=str, default='./models_local/TRresNet_L_448_86.6.pth')
parser.add_argument('--pic_path', type=str, default='./pics/000000000885.jpg')
parser.add_argument('--model_name', type=str, default='tresnet_l')
parser.add_argument('--validation_data_file', type=str, default='validation_filter.csv')
parser.add_argument('--checkpoint_path', type=str, required=True)

parser.add_argument('--input_size', type=int, default=448)
parser.add_argument('--dataset_type', type=str, default='MS-COCO')
parser.add_argument('--th', type=float, default=0.6)
parser.add_argument('--label', type=str, default="glass")
parser.add_argument('--visualize_err', action="store_true")
parser.add_argument('--whitelist', action="store_true")
parser.add_argument('--distr', action="store_true")
parser.add_argument('--exact', action="store_true")
parser.add_argument('--limit', type=int, default=15)
parser.add_argument('--num_eval', type=int, default=5000)

parser.add_argument('--gamma_neg', type=int, default=5)
parser.add_argument('--gamma_pos', type=int, default=5)

parser.add_argument('--top', type=int, default=15)
parser.add_argument('--wl_path', type=str, default="recycle")
parser.add_argument('--split', type=str, default="recycle")
parser.add_argument('--app', type=str, default="aanetl")
parser.add_argument('--app_type', type=str, default="multi_choice")
parser.add_argument('--method', type=str, default=None)
parser.add_argument('--w', type=str, default="recycle")

args = parse_args(parser)

whitelist_mapping = {}
wl_mapping = pd.read_csv(args.wl_path)
all_classes_count = {}

for i in range(len(wl_mapping)):
    if wl_mapping.iloc[i]['wl'] not in whitelist_mapping:
        whitelist_mapping[wl_mapping.iloc[i]['wl']] = []
        all_classes_count[wl_mapping.iloc[i]['wl']] = 0
    whitelist_mapping[wl_mapping.iloc[i]['wl']].append(wl_mapping.iloc[i]['class_name'])
mapping_dict = whitelist_mapping
priority_list = list(whitelist_mapping.keys())
priority_list.append("other")
print(priority_list)
print(whitelist_mapping)
# priority_list.append("other")

def read_stats(data_path, wl_path, mid_to_human_class):
    data_file = pd.read_csv(data_path)
    stats_dict= {}
    all_whitelist_classes = []
    for key in wl_path:
        stats_dict[key] = 0
        all_whitelist_classes.extend(wl_path[key])
    stats_dict['other'] = 0
    
    for index in range(len(data_file)):
        all_classes = data_file.iloc[index]['class_list']
        all_classes = all_classes.split('[')[1].split(']')[0].replace("'", "").split(', ')
        all_classes_human_read = [mid_to_human_class[i] for i in all_classes]
        all_classes_neg = data_file.iloc[index]['class_list_neg']
        all_classes_neg = all_classes_neg.split('[')[1].split(']')[0].replace("'", "").split(', ')

        all_neg_classes_human_read =  []
        if all_classes_neg[0] != '':
            all_neg_classes_human_read = [mid_to_human_class[i] for i in all_classes_neg]
        if len(intersection(all_whitelist_classes, all_classes_human_read)) > 0:
            for key in wl_path:
                if len(intersection(wl_path[key], all_classes_human_read)) > 0:
                    stats_dict[key] += 1
        if len(intersection(all_whitelist_classes, all_neg_classes_human_read)) > 0:
            stats_dict['other'] += 1
    return stats_dict
def define_whitelist_wl(all_classes):
    
    
    existing_classes = []
    existing_classes_length = []
    for class_name in mapping_dict:
        if len(intersection(mapping_dict[class_name], all_classes)) > 0:
            existing_classes.append(class_name)
            existing_classes_length.append(len(intersection(mapping_dict[class_name], all_classes)))
    # print(all_classes)
    # print("WL gt", existing_classes)

    return list(set(existing_classes)), existing_classes_length
def tranform(label):
   
    finer_class_to_wl_mapping = {}
    for cls_name in mapping_dict:
       
        for finest_class in mapping_dict[cls_name]:
            finer_class_to_wl_mapping[finest_class] = cls_name
    if label not in finer_class_to_wl_mapping:
        return label
    return finer_class_to_wl_mapping[label]

def calculate_precision_recall_i(args, conf_matrix, image_name, error_dict_file, detected_classes_sorted, gt_classes, gt_wl_neg, result_dict,result_dict_neg, negative_fp, priority_list, donate_num, gt_raw_classes):
    predicted_wl = None
    # print(detected_classes_sorted)
    detected_classes_sorted = [tranform(label) for label in detected_classes_sorted]
    # print(detected_classes_sorted)
    # assert len(intersection(gt_classes, gt_wl_neg)) == 0
    for label in detected_classes_sorted:
        if label not in priority_list: continue
        label_index = list(priority_list).index(label)
        predicted_wl = label
        break
    # if len(gt_classes) > 1:
    #     print(predicted_wl)
    #     print(gt_classes)
    #     print("=============")
    if len(gt_classes) == 0:
        conf_gt_class = 'other'
    else:
        conf_gt_class = gt_classes[0]
        
    if predicted_wl is None:

        conf_matrix[conf_gt_class][len(priority_list)-1] += 1
    else:
        conf_matrix[conf_gt_class][priority_list.index(predicted_wl)] += 1
    if len(gt_classes) == 0:
        if  predicted_wl is None:
            result_dict['acc'][-1] += 1
            return False
        else:
            result_dict['fp'][priority_list.index(predicted_wl)] += 1
            return True
   
    if predicted_wl:
        predicted_label_index = list(priority_list).index(predicted_wl)
        if predicted_wl in gt_classes:
            result_dict['acc'][priority_list.index(gt_classes[0])] += 1
            if len(gt_classes) > 1 and predicted_wl != gt_classes[0]:
                result_dict['solv'] += 1
            for gt in gt_classes:
                gt_label_index = list(priority_list).index(gt)
                result_dict['tp'][gt_label_index] += 1
                
            return False
        else:
            result_dict['fn'][priority_list.index(gt_classes[0])] += 1
            if predicted_wl in gt_wl_neg:
                result_dict['fp'][predicted_label_index] += 1
                
            
    else:
        result_dict['fp'][-1] += 1
        result_dict['fn'][priority_list.index(gt_classes[0])] += 1
    return True


def calculate_precision_recall_r(args, image_name, error_dict_file, gt_raw_neg, detected_classes_sorted, \
gt_classes, gt_wl_neg, result_dict,result_dict_neg, negative_fp, priority_list, donate_num, gt_raw_classes):
    predicted_wl = None
    # print(detected_classes_sorted)
    # detected_classes_sorted = [tranform(label) for label in detected_classes_sorted]
    # print(detected_classes_sorted)
    fine_predicted_label = None

    # assert len(intersection(gt_classes, gt_wl_neg)) == 0
    for label in detected_classes_sorted:
        wl_label = tranform(label)
        if wl_label not in priority_list: continue
        # if label_index < 3:
        #     predicted_wl = label
        #     break
        
        predicted_wl = wl_label
        fine_predicted_label = label
        break
   
    if len(gt_classes) == 0:
        if fine_predicted_label not in gt_raw_neg:
        # if predicted_wl not in gt_wl_neg:
            result_dict['acc'][-1] += 1
            result_dict['tp'][-1] += 1
        else:
            result_dict['fp'][priority_list.index(predicted_wl)] += 1
        return
        
    if predicted_wl:
        predicted_label_index = list(priority_list).index(predicted_wl)
        
        if fine_predicted_label not in gt_raw_neg:
        # if predicted_wl not in gt_wl_neg:
            for gt in gt_classes:
                gt_label_index = list(priority_list).index(gt)
                result_dict['tp'][gt_label_index] += 1
                result_dict['acc'][gt_label_index] += 1
                break
        else:
            # print("GT classes: ", gt_classes)
            # print(predicted_wl)
            for neg in gt_classes:
                result_dict['fn'][priority_list.index(neg)] += 1
            if fine_predicted_label in gt_raw_neg:
                result_dict['fp'][predicted_label_index] += 1
                
            
    else:
        result_dict['fp'][-1] += 1
        for gt in gt_classes:
            result_dict['fn'][priority_list.index(gt)] += 1
def whitelist_index(item):
    for key in mapping_dict:
        all_lower = [i.lower() for i in mapping_dict[key]]
        if item in all_lower:
            return key

    return item

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))
def calculate_avg(priority_recall_count, all_classes_count):
    all_acc = 0
    non_zeros = 0
    for key in priority_recall_count:
        if all_classes_count[key] > 0:
            non_zeros += 1
            all_acc += priority_recall_count[key] / all_classes_count[key]
    if non_zeros == 0: return 0
    return all_acc / non_zeros


def calculate_avg_loss(priority_recall_count, all_classes_count):
    all_acc = 0
    non_zeros = 0
    for key in priority_recall_count:
        if all_classes_count[key] > 0:
            non_zeros += 1
            all_acc += sum(priority_recall_count[key]) / all_classes_count[key]
    return all_acc / non_zeros

def parse_input(all_classes, mid_to_human_class, class_list, W):
    output = torch.zeros((1, len(class_list)), dtype=torch.float32)
    for class_name in all_classes:
        if class_name == '':
            continue
        if mid_to_human_class[class_name].replace("'", "") in class_list:
            pos_class_idx_orig = class_list.index(mid_to_human_class[class_name].replace("'", "")) # The original index in class list
            output[0][pos_class_idx_orig] = 1


    return output
def define_whitelist(all_classes):

    existing_classes = []
    for class_name in all_classes:
        for label in mapping_dict:
            if class_name in mapping_dict[label]:
                existing_classes.append(label)
    return existing_classes

def main():

    # parsing args
    exact_labels = [args.label]

    class_list_file = pd.read_csv("all_classes.csv")
    state = torch.load(args.model_path, map_location='cpu')

    class_list = list(state['idx_to_class'].values())
    class_list = [i.replace('\"', "") for i in class_list]
    class_list = [i.replace("'", "") for i in class_list]
    W = []
    W_human = []
    sub_classes_list = []
    for key in mapping_dict:
        for value in mapping_dict[key][:10]:
            if value.lower() in class_list:
                W.append(class_list.index(value.lower()))
                sub_classes_list.append(value)
                W_human.append(value)
    # Setup model

    args.num_classes = len(class_list)
    args.do_bottleneck_head = True
    model = create_model(args).cuda()

    sub_classes_list = np.array(class_list)
    if args.distr:
        model = torch.nn.DataParallel(model)

    model.cuda()
    model.eval()
    wrong = 0
    wrong_relaxed = 0
    # doing inference
 
    human_verified_images = list(pd.read_csv("test-images-with-rotation.csv")["ImageID"])
    validation_data_mapping = {}
    validation_data_file = pd.read_csv(args.validation_data_file)
    validation_to_class_mapping = {}

    mid_to_human_class_file = pd.read_csv("oidv6-class-descriptions.csv")
    mid_to_human_class = {}
    for i in range(len(mid_to_human_class_file)):
        mid_to_human_class[mid_to_human_class_file.iloc[i][0]] = mid_to_human_class_file.iloc[i][1]


    stats = read_stats(args.validation_data_file, whitelist_mapping, mid_to_human_class)
    wl_validation_data_mapping = {}
    validation_data_mapping_neg = {}
    wl_validation_data_mapping_neg = {}
    sample_indices = np.arange(len(validation_data_file))
    np.random.seed(42)
    np.random.shuffle(sample_indices)
    count = 0
    training_images = []
    for idx in range(len(validation_data_file)):
        index = sample_indices[idx]
        if validation_data_file.iloc[index]['split'] != args.split: 
            training_images.append(validation_data_file.iloc[index]['image_id'])
    for idx in range(len(validation_data_file)):
        
        count += 1
        index = sample_indices[idx]
        if validation_data_file.iloc[index]['split']  != args.split: 
            continue
        if validation_data_file.iloc[index]['image_id'] in training_images:
            continue
        file_name = validation_data_file.iloc[index]['image_id'].split("/")[-1]
        all_classes = validation_data_file.iloc[index]['class_list']
        all_classes = all_classes.split('[')[1].split(']')[0].replace("'", "").split(', ')

        validation_to_class_mapping[file_name] = parse_input(all_classes, mid_to_human_class, class_list, W)
        all_classes = [mid_to_human_class[i] for i in all_classes]
        validation_data_mapping[file_name] = define_whitelist(all_classes)
        wl_validation_data_mapping[file_name] = all_classes


        all_neg_classes = validation_data_file.iloc[index]['class_list_neg']
        all_neg_classes = all_neg_classes.split('[')[1].split(']')[0].replace("'", "").split(', ')
        validation_data_mapping_neg[file_name] = parse_input(all_neg_classes, mid_to_human_class, class_list, W)

        if all_neg_classes[0] != '':
            all_neg_classes = [mid_to_human_class[i] for i in all_neg_classes]
        else:
            all_neg_classes = []
        wl_validation_data_mapping_neg[file_name] = all_neg_classes


        # print(validation_to_class_mapping[file_name])

    ckpt_name = args.checkpoint_path
    all_classes_count = {}
    exact_recall_count = {}
    relaxed_recall_count = {}
    priority_recall_count = {}
    loss_value_dict = {}
    all_images = 0

    for i in mapping_dict:
        all_classes_count[i] = 0
        exact_recall_count[i] = 0
        relaxed_recall_count[i] = 0
        priority_recall_count[i] = 0
        loss_value_dict[i] = []
    
    model.load_state_dict(torch.load(ckpt_name, map_location='cpu'), strict=True)

    model.cuda()
    model.eval()
    all_images_count = {}
    all_images_count_neg = {}
    for key in priority_list:
        all_images_count[key] = 0
        all_images_count_neg[key] = 0
    result_dict_r = {'acc': [0 for _ in range(len(priority_list) )], 'fp': [0 for _ in range(len(priority_list) )], 'fn': [0 for _ in range(len(priority_list) )], 'tp': [0 for _ in range(len(priority_list) )]}
    result_dict = {'solv': 0, 'acc': [0 for _ in range(len(priority_list) )], 'fp': [0 for _ in range(len(priority_list) )], 'fn': [0 for _ in range(len(priority_list) )], 'tp': [0 for _ in range(len(priority_list) )]}
    result_dict_neg = {'acc': [0 for _ in range(len(priority_list) )], }
    negative_fp = {'neg_fp': 0}
    class_list = np.array(class_list)
    donate_num = {0: 0}
    all_loss = []

    wl_loss = []
    error_dict_file = {"image_id": [],  'error': []}
    conf_matrix = {}
    for key in priority_list:
        conf_matrix[key] = [0 for i in range(len(priority_list))] # Key is ground truth
    gt_ratio = {}
    for key in mapping_dict:
        gt_ratio[key] = []
    all_results = np.zeros((len(validation_data_mapping), 9605 + len(priority_list)))
    for image_name in validation_data_mapping:
        gt_wl, gt_length = define_whitelist_wl(wl_validation_data_mapping[image_name])
        gt_wl_neg = define_whitelist_wl(wl_validation_data_mapping_neg[image_name])
        if len(set(gt_wl)) > 0:
            for label in set(gt_wl):

                all_images_count[label] += 1
        else:
            all_images_count['other'] += 1

        if "jpg" not in image_name: continue
        image = os.path.join(args.pic_path, image_name)

        im = Image.open(image)
        im_resize = im.resize((args.input_size, args.input_size))
        np_img = np.array(im_resize, dtype=np.uint8)

        try:
            tensor_img = torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0  # HWC to CHW
            tensor_batch = torch.unsqueeze(tensor_img, 0).cuda()
            start_time = time.time()

            with torch.no_grad():
                with autocast():
                    output_orig = model(tensor_batch).float()
                    
            torch.cuda.synchronize()
            output = torch.squeeze(torch.sigmoid(output_orig))


        except:
            continue
        target = validation_to_class_mapping[image_name]
        target = target.cuda()
        target_neg = validation_data_mapping_neg[image_name]
        target_neg = target_neg.cuda()
        # print(output.shape)
        np_output = output.cpu().detach().numpy()

        # Cache results
        all_results[all_images][:9605] = np_output
        if len(gt_wl) == 0:

            all_results[all_images][-1] = 1
        else:
            for gt in gt_wl:
                all_results[all_images][priority_list.index(gt) + 9605] = 1



        top_k_th = max(np.sort(np_output)[-10], args.th)
        detected_classes = class_list[(np_output >= top_k_th)]
        np_indices = np.argsort(np_output)[::-1][:args.top]
        detected_classes_sorted = []
        for np_index in np_indices:
            if np_output[np_index] > args.th:
                detected_classes_sorted.append(class_list[np_index])

        detected_classes = [i.lower().replace("'", '').split()[0] for i in detected_classes]
        all_images += 1
        wrong = calculate_precision_recall_i(args, conf_matrix, image_name, error_dict_file, detected_classes_sorted, gt_wl, gt_wl_neg, result_dict,result_dict_neg, negative_fp, priority_list, donate_num, wl_validation_data_mapping[image_name])
        calculate_precision_recall_r(args, image_name, error_dict_file, wl_validation_data_mapping_neg[image_name], detected_classes_sorted, gt_wl, gt_wl_neg, result_dict_r,result_dict_neg, negative_fp, priority_list, donate_num, wl_validation_data_mapping[image_name])
        overall_acc = 0
        
    overall_acc = sum(result_dict['acc']) / all_images
    print("Overall acc: ", overall_acc)
    print("app name: ", args.wl_path)
    if f"results_{args.app_type}_{args.method}.csv" not in os.listdir("results"):
        df = pd.DataFrame({"app": [args.app], "precision": [overall_acc]})
        df.to_csv(f"results/results_{args.app_type}_{args.method}.csv", index=False)
    else:
        df = pd.read_csv(f"results/results_{args.app_type}_{args.method}.csv")
        df2 = pd.DataFrame({"app": [args.app], "precision": [overall_acc]})
        df = pd.concat([df, df2], ignore_index=True)
        df.to_csv(f"results/results_{args.app_type}_{args.method}.csv", index=False)
if __name__ == '__main__':
    main()
