from builtins import breakpoint
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
import os
from torch.cuda.amp import GradScaler, autocast

parser = argparse.ArgumentParser(description='ASL MS-COCO Inference on a single image')
parser.add_argument('--saved_model_path', type=str, default='./models/model-highest.ckpt')

parser.add_argument('--model_path', type=str, default='./models_local/TRresNet_L_448_86.6.pth')
parser.add_argument('--pic_path', type=str, default='./pics/000000000885.jpg')
parser.add_argument('--model_name', type=str, default='tresnet_l')
parser.add_argument('--validation_data_file', type=str, default='validation_filter.csv')
parser.add_argument('--app', type=str, default="aanetl")
parser.add_argument('--app_type', type=str, default="multi_choice")
parser.add_argument('--method', type=str, default=None)

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
parser.add_argument('--checkpoint_path', type=str, default="recycle")
parser.add_argument('--lower_bound', type=int, default=0)
parser.add_argument('--upper_bound', type=int, default=2000000)
parser.add_argument('--epoch_lower_bound', type=int, default=0)
parser.add_argument('--epoch_upper_bound', type=int, default=30)
parser.add_argument('--alpha', type=int, default=7)
parser.add_argument('--alpha1', type=float, default=7)
parser.add_argument('--alpha_other', type=int, default=7)
parser.add_argument('--alpha3', type=float, default=7)

parser.add_argument('--alpha2', type=int, default=2)
parser.add_argument('--focal', action="store_true")
parser.add_argument('--asymm', action="store_true")
parser.add_argument('--priority', action="store_true")
parser.add_argument('--penalize_other', action="store_true")
parser.add_argument('--optimize', action="store_true")
parser.add_argument('--final', action="store_true")
parser.add_argument('--verbose', action="store_true")
parser.add_argument('--sigmoid', action="store_true")

parser.add_argument('--weight_balancing', action="store_true")

parser.add_argument('--wl_path', type=str, default="recycle")
parser.add_argument('--frac', type=str, default="recycle")
parser.add_argument('--top', type=int, default=15)
parser.add_argument('--split', type=str, default="test")

parser.add_argument('--w', type=str, default="recycle")

args = parse_args(parser)

whitelist_mapping = {}
wl_mapping = pd.read_csv(args.wl_path)
all_classes_count = {}
index_dict = [[6176, 6171, 6173, 6175, 4303, 8685, 8703], [6514, 1189, 6512]]
for i in range(len(wl_mapping)):
    if wl_mapping.iloc[i]['wl'] not in whitelist_mapping:
        whitelist_mapping[wl_mapping.iloc[i]['wl']] = []
        all_classes_count[wl_mapping.iloc[i]['wl']] = 0
    whitelist_mapping[wl_mapping.iloc[i]['wl']].append(wl_mapping.iloc[i]['class_name'])
mapping_dict = whitelist_mapping
priority_list = list(whitelist_mapping.keys())
print(priority_list)
print(whitelist_mapping)
priority_list.append("other")

def define_whitelist_wl(all_classes):
    
    
    existing_classes = []
    for class_name in mapping_dict:
        if len(intersection(mapping_dict[class_name], all_classes)) > 0:
            existing_classes.append(class_name)
    # print(all_classes)
    # print("WL gt", existing_classes)

    return list(set(existing_classes))
def tranform(label):
   
    finer_class_to_wl_mapping = {}
    for cls_name in mapping_dict:
       
        for finest_class in mapping_dict[cls_name]:
            finer_class_to_wl_mapping[finest_class] = cls_name
    if label not in finer_class_to_wl_mapping:
        return label
    return finer_class_to_wl_mapping[label]
def calc_loss(x1, x2, margin, alpha2, alpha3):
    x1 = torch.Tensor([x1])
    x2 = torch.Tensor([x2])
    if x2-x1+margin > 0:
        return alpha2 * 1/(1+torch.exp(-alpha3 * (x2-x1+margin)))
    return 1/(1+torch.exp(-alpha3 * (x2-x1+margin)))

def calculate_precision_recall(np_output, loss_dict, args, image_name, detected_classes_sorted, gt_classes, gt_wl_neg, result_dict,result_dict_neg,priority_list, gt_raw_classes):
    predicted_wl = []
    detected_classes_sorted = [tranform(label) for label in detected_classes_sorted]
    for label in detected_classes_sorted:
        if label not in priority_list: continue
        label_index = list(priority_list).index(label)
        if label_index < len(priority_list) :
            predicted_wl.append(label)
    wrong_flag = False
    if len(intersection(priority_list, predicted_wl)) != len(gt_classes):
        wrong_flag = True

    if wrong_flag is False:
        if len(gt_classes) == 0:
            result_dict['tp'][-1] += 1
        else:
            for gt in gt_classes:
                result_dict['tp'][priority_list.index(gt)] += 1
        result_dict['acc'] += 1
    



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
        if validation_data_file.iloc[index]['split'] != args.split:
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
    
    model.load_state_dict(torch.load(os.path.join( ckpt_name), map_location='cpu'), strict=True)

    model.cuda()
    model.eval()
    all_images_count = {}
    all_images_count_neg = {}
    loss_dict = []
    for key in priority_list:
        all_images_count[key] = 0
        all_images_count_neg[key] = 0
    result_dict_neg = {'tp': [0 for _ in range(len(priority_list))], 'fp': [0 for _ in range(len(priority_list))], 'fn': [0 for _ in range(len(priority_list))]}

    result_dict = {'acc': 0, 'tp': [0 for _ in range(len(priority_list))], 'fp': [0 for _ in range(len(priority_list))], 'fn': [0 for _ in range(len(priority_list))]}

    for image_name in validation_data_mapping:
        gt_wl = define_whitelist_wl(wl_validation_data_mapping[image_name])
        gt_wl_neg = define_whitelist_wl(wl_validation_data_mapping_neg[image_name])
        
        
        for label in set(gt_wl):

            all_images_count[label] += 1
        
        if len(gt_wl) == 0:
            all_images_count['other'] += 1
        if len(set(gt_wl)) == 0 and len(set(gt_wl_neg)) == 0:
            raise Exception('error')
        if "jpg" not in image_name: continue
        image = os.path.join(args.pic_path, image_name)

        im = Image.open(image)
        im_resize = im.resize((args.input_size, args.input_size))
        np_img = np.array(im_resize, dtype=np.uint8)


        try:
            tensor_img = torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0  # HWC to CHW
            tensor_batch = torch.unsqueeze(tensor_img, 0).cuda()

            with torch.no_grad():
                with autocast():
                    output_orig = model(tensor_batch).float()
            output = torch.squeeze(torch.sigmoid(output_orig))


        except:
            continue
        target = validation_to_class_mapping[image_name]
        target = target.cuda()
        target_neg = validation_data_mapping_neg[image_name]
        target_neg = target_neg.cuda()
        
        np_output = output.cpu().detach().numpy()
        np_indices = np.argsort(np_output)[::-1][:args.top]
        detected_classes_sorted = []
        for np_index in np_indices:
            if np_output[np_index] > args.th:
                detected_classes_sorted.append(class_list[np_index])
        all_images += 1
        calculate_precision_recall(np_output, loss_dict, args, image_name, \
            detected_classes_sorted, gt_wl, gt_wl_neg, result_dict,result_dict_neg,\
                priority_list, wl_validation_data_mapping[image_name])

    
    acc = float(result_dict['acc'] / all_images)
    
    print("Overall acc: ", acc)
    if f"results_{args.app_type}_{args.method}.csv" not in os.listdir("results"):
        df = pd.DataFrame({"app": [args.app], "precision": [acc]})
        df.to_csv(f"results/results_{args.app_type}_{args.method}.csv", index=False)
    else:
        df = pd.read_csv(f"results/results_{args.app_type}_{args.method}.csv")
        df2 = pd.DataFrame({"app": [args.app], "precision": [acc]})
        df = pd.concat([df, df2], ignore_index=True)
        df.to_csv(f"results/results_{args.app_type}_{args.method}.csv", index=False)
if __name__ == '__main__':
    main()