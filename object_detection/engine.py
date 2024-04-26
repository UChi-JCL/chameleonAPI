from builtins import breakpoint
import math
import sys
import time
import torch

import torchvision
from dataset.coco_utils import get_coco_api_from_dataset
from dataset.coco_eval import CocoEvaluator

import utils
import pandas as pd
from predict import *
import cv2
from PIL import Image
import os
import pandas as pd
def transform(raw_input, mapping_dict):
    inputs_n = torch.zeros((len(raw_input), len(mapping_dict)))
    for i in range(len(raw_input)):
        for j in range(len(raw_input[i])):
            if raw_input[i]['labels'][j] in mapping_dict['wl1']:
                inputs_n[i][0] += 1
            else:
                inputs_n[i][1] += 1
    return inputs_n

def map_orig_coco_labels_to_wl_labels(labels, mapping_dict):
    new_labels = []
    _, _, _, coco_index_to_name_mapping = get_coco_label_names()
    all_wl_names =list( mapping_dict["wl1"])
    for label in labels:
        if label not in all_wl_names:
            new_labels += [len(all_wl_names)]
        else:
            label_index = all_wl_names.index(label)
            new_labels += [label_index]
    return new_labels
def new_mapping(labels, mapping_dict):
    new_labels = []
    for label in labels:
        if label in mapping_dict['wl1']:
            new_labels += [0]
        else:
            new_labels += [1]
    return new_labels
def train_one_epoch(args, model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
    count = 0
    def transform_gt(label, mapping_dict):
        if label.item() in mapping_dict['wl1']: return 1
        return 2
    mapping_dict = read_whitelist(args)
    for images, targets, _ in metric_logger.log_every(data_loader, print_freq, header):
        count += 1
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        if "CC" in os.environ and os.environ["CC"] == "True":
            
            for i in range(len(targets)):
                targets[i]["labels"] = torch.tensor([transform_gt(label, mapping_dict) for label in targets[i]["labels"]]).long().cuda()
        elif os.environ["WL_EVAL"] == "True":
            for i in range(len(targets)):
                targets[i]["labels"] = torch.tensor(map_orig_coco_labels_to_wl_labels(targets[i]["labels"].cpu().detach().numpy(),\
                    mapping_dict)).to(device).long()
        elif os.environ["NEW"] == "True":
            for i in range(len(targets)):
                targets[i]["labels"] = torch.tensor(new_mapping(targets[i]["labels"].cpu().detach().numpy(),\
                    mapping_dict)).to(device).long()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
    
        model.train()
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # if args.output_dir and count % 1000 == 0:
        #     model_without_ddp = model
        #     utils.save_on_master({
        #         'model': model_without_ddp.state_dict(),
        #         'args': args,
        #         'epoch': epoch},
        #         os.path.join(args.output_dir, f'model_{epoch}_{count}.pth'))
          
def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for image, targets, _ in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(image)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator

def read_whitelist(args):
    whitelist_mapping = {}
    wl_mapping = pd.read_csv(args.wl_path)
    all_classes_count = {}
    coco_label_names, coco_class_ids, coco_name_to_index_mapping, coco_index_to_name_mapping = get_coco_label_names()
    for i in range(len(wl_mapping)):
        if wl_mapping.iloc[i]['wl'] not in whitelist_mapping:
            whitelist_mapping[wl_mapping.iloc[i]['wl']] = []
            all_classes_count[wl_mapping.iloc[i]['wl']] = 0
        whitelist_mapping[wl_mapping.iloc[i]['wl']].append(coco_name_to_index_mapping[wl_mapping.iloc[i]['class_name']])
    mapping_dict = whitelist_mapping
    mapping_dict['other'] = []
    return mapping_dict


def transform_baseline_cls_to_coco(baseline_idx, mapping_dict, mapping_dict_test):
    if baseline_idx.item() == len(mapping_dict["wl1"]): return -1
    label_name = mapping_dict["wl1"][baseline_idx]
    return label_name
    
def read_whitelist_test(args):
    whitelist_mapping = {}
    wl_mapping = pd.read_csv(args.wl_path_test)
    all_classes_count = {}
    coco_label_names, coco_class_ids, coco_name_to_index_mapping, coco_index_to_name_mapping = get_coco_label_names()
    for i in range(len(wl_mapping)):
        if wl_mapping.iloc[i]['wl'] not in whitelist_mapping:
            whitelist_mapping[wl_mapping.iloc[i]['wl']] = []
            all_classes_count[wl_mapping.iloc[i]['wl']] = 0
        whitelist_mapping[wl_mapping.iloc[i]['wl']].append(coco_name_to_index_mapping[wl_mapping.iloc[i]['class_name']])
    mapping_dict = whitelist_mapping
    mapping_dict['other'] = []
    return mapping_dict

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))
def find_gt_wl(label, mapping_dict):
    
    for key in mapping_dict:
        if  label in mapping_dict[key]:
            return key
    
    return 'other'

def find_gt_wl_baseline(label, test_mapping_dict):
    """
    mapping_dict: the wl mapping dict used to TRAIN the new baselines 
    test_mapping_dict: the wl mapping dict used to TEST (from the apps)
    """
    for key in test_mapping_dict:
        if label in test_mapping_dict[key]:
            return key
    return "other"


def evaluate_sw_old(args, model, dataset, device, root):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    cpu_device = torch.device("cpu")
    coco = get_coco_api_from_dataset(dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)
    coco_evaluator.coco_eval['bbox'].params.useCats = 0
    mapping_dict = read_whitelist(args)
    priority_list = list(mapping_dict.keys())
    result_dict_bbox_ag = {'fp': [0 for _ in range(len(priority_list) )], 'fn': [0 for _ in range(len(priority_list) )], 'tp': [0 for _ in range(len(priority_list) )]}

    result_dict = {'fp': [0 for _ in range(len(priority_list) )], 'fn': [0 for _ in range(len(priority_list) )], 'tp': [0 for _ in range(len(priority_list) )]}
    count = 0
    
    
    acc = 0
    avg_acc = []
    all_wl_count = 0 
    all_others_count = 0
    acc_dict = {}
    images_count = {}
    for key in mapping_dict:
        acc_dict[key] = []
    o_count = 0
    mapping_dict_test = read_whitelist_test(args)
    f1_list, recall_list, prec_list, acc_list = [], [], [], []
    tp, fp, fn = 0, 0, 0
    tp_o, fp_o, fn_o = 0, 0, 0
    for (img, anno, image_id) in dataset:
        count += 1
        o_count = 0
        acc = 0
        others_count = 0
        others_acc = 0
        if count > 500: break
        if count % 50 == 0:
            if tp == 0:
                re = 0
                prec = 0
                f1 = 0
            else:
                re = tp / (tp + fn)
                prec = tp / (tp + fp)
                f1 = 2 * (re * prec) / (re + prec)
                
            if tp_o == 0:
                re_o = 0
                prec_o = 0
                f1_o = 0
            else:
                re_o = tp_o / (tp_o + fn_o)
                prec_o = tp_o / (tp_o + fp_o)
                f1_o = 2 * (re_o * prec_o) / (re_o + prec_o)
            

        path = coco.loadImgs(image_id)[0]["file_name"]
        image_raw = cv2.imread(os.path.join(root, path))
        result, output, top_predictions = predict(image_raw, model, device, mapping_dict)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in output]
        res = {int(image_id): outputs[0]}
        coco_evaluator.update(res)
        ious = coco_evaluator.coco_eval['bbox'].computeIoU(image_id, 0)
        pred_wls = []
        gt_wls =  []
        pred_labels = output[0]['labels']
        pred_scores = output[0]['scores']
        gt_wls = []
        try:
            for cls_ind in range(len(anno['labels'])):
                
                if os.environ["WL_EVAL"] == "True":
                    gt_wl = find_gt_wl_baseline(anno['labels'][cls_ind], mapping_dict_test)
                else:
                    gt_wl = find_gt_wl_baseline(anno['labels'][cls_ind], mapping_dict)
                gt_wls += [gt_wl]
                if gt_wl != 'other':
                    o_count += 1
                    all_wl_count += 1
                else:
                    all_others_count += 1
                    others_count += 1
        except:
            pass
        pred_wls = []
        for i in range(len(pred_labels)):
            if os.environ["WL_EVAL"] == "True":
                if os.environ["CC"] == "False":
                    cls_name = transform_baseline_cls_to_coco(pred_labels[i], mapping_dict, mapping_dict_test)
                    pred_wl = find_gt_wl_baseline(cls_name, mapping_dict_test)
                    pred_wls += [pred_wl]
                
            elif os.environ['NEW'] == "True":
                if pred_labels[i] == 1:
                    pred_wl = 'wl1'
                else:
                    pred_wl = 'other'
                    
            else:
                pred_wl = find_gt_wl_baseline(pred_labels[i], mapping_dict)
            if pred_wl != 'other' \
                and pred_scores[i] > args.th:
                acc += 1
            elif pred_wl == 'other' and pred_scores[i] > args.th:
                others_acc += 1
            # elif pred_scores[i] > 0 and pred_wl == 'other':
            #     acc['other'] += 1
        if o_count == 0: continue
        tp += min(acc, o_count)
        fp += max(0, acc - o_count)
        fn += max(0, o_count - acc)
        
        tp_o += min(others_acc, others_count)
        fp_o += max(0, others_acc - others_count)
        fn_o += max(0, others_count - others_acc)

        if o_count > 0 and others_count > 0:
            acc_list.append(((1 - abs(o_count-acc)/o_count  ) + (1-abs(others_count-others_acc)/others_count)) / 2)
        # assert f1 >= 0 and f1 <= 1
    print("F1: ", f1)
    return f1


def evaluate_sw(args, model, dataset, device, root):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    cpu_device = torch.device("cpu")
    coco = get_coco_api_from_dataset(dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)
    coco_evaluator.coco_eval['bbox'].params.useCats = 0
    mapping_dict = read_whitelist(args)
    mapping_dict_test = read_whitelist_test(args)
    priority_list = list(mapping_dict.keys())
    result_dict_bbox_ag = {'fp': [0 for _ in range(len(priority_list) )], 'fn': [0 for _ in range(len(priority_list) )], 'tp': [0 for _ in range(len(priority_list) )]}

    result_dict = {'fp': [0 for _ in range(len(priority_list) )], 'fn': [0 for _ in range(len(priority_list) )], 'tp': [0 for _ in range(len(priority_list) )]}
    count = 0
    print("lenght of data: ", len(dataset))
    
    
    acc = 0
    avg_acc = []
    acc_dict = {}
    for key in mapping_dict:
        acc_dict[key] = []
    # available_ids = list(pd.read_csv("avail_split7_data.csv")['image_id'])
    tp, fp, fn, acc = {}, {}, {}, {}
    all_count = {}
    acc_list = {}
    for key in mapping_dict:
        # if key == "other": continue
        tp[key], fp[key], fn[key] = 0, 0, 0
        acc_list[key] = []
    for (img, anno, image_id) in dataset:
        # if image_id not in available_ids:
        #     continue
        count += 1
        if count > 500: break
        all_count = {}
        acc = {}
        for key in mapping_dict:
            all_count[key] = 0
            acc[key] = 0
            
        if count % 50 == 0:
            f1_avg = []
            for key in mapping_dict:
                re, prec, f1 = 0, 0, 0
                if tp[key] == 0:
                    re = 0
                    prec = 0
                    f1 = 0
                else:
                    re = tp[key] / (tp[key] + fn[key])
                    prec = tp[key] / (tp[key] + fp[key])
                    f1 = 2 * (re * prec) / (re + prec)
                f1_avg += [f1]
            print("F1: ", np.mean(f1_avg))
            avg_acc = []
            for key in mapping_dict:
                avg_acc += [np.mean(acc_list[key])]
                
            print("Accuracy", np.mean(avg_acc))
            # print(o_f1)
        path = coco.loadImgs(image_id)[0]["file_name"]
        image_raw = cv2.imread(os.path.join(root, path))
        
        result, output, top_predictions = predict(image_raw, model, device, mapping_dict)
        cv2.imwrite(f'outputs/pred_{image_id}.jpg', result)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in output]
        res = {int(image_id): outputs[0]}
        coco_evaluator.update(res)
        ious = coco_evaluator.coco_eval['bbox'].computeIoU(image_id, 0)
        pred_wls = []
        gt_wls =  []
        pred_labels = output[0]['labels']
        pred_scores = output[0]['scores']
        
        try:
            for cls_ind in range(len(anno['labels'])):
                if os.environ["WL_EVAL"] == "True":
                    gt_wl = find_gt_wl_baseline(anno['labels'][cls_ind], mapping_dict_test)
                else:
                    gt_wl = find_gt_wl_baseline(anno['labels'][cls_ind], mapping_dict)
                all_count[gt_wl] += 1
        except:
            pass
        for i in range(len(pred_labels)):
            
            if os.environ["WL_EVAL"] == "True":
                if os.environ["CC"] == "False":
                    cls_name = transform_baseline_cls_to_coco(pred_labels[i], mapping_dict, mapping_dict_test)
                    pred_wl = find_gt_wl_baseline(cls_name, mapping_dict_test)
                else:
                    if pred_labels[i] == 1:
                        pred_wl = 'wl1'
                    elif pred_labels[i] == 2:
                        pred_wl = 'other'
            else:
                pred_wl = find_gt_wl_baseline(pred_labels[i], mapping_dict)
            if pred_wl != 'other' \
                and pred_scores[i] > args.th:
                acc[pred_wl] += 1
            elif pred_scores[i] > 0 and pred_wl == 'other':
                acc['other'] += 1
        for key in mapping_dict:
            # if key == "other": continue 
            
            tp[key] += min(acc[key], all_count[key])
            fp[key] += abs( acc[key] - all_count[key])
            fn[key] += abs(all_count[key] - acc[key])
        for key in mapping_dict:
            # if key == "other": continue
            if all_count[key] > 0:
                acc_list[key].append(max(0, 1 - abs(all_count[key] - acc[key])/all_count[key]))

