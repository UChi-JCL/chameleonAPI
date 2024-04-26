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
from PIL import Image
import os
import pandas as pd
# def transform(raw):
#     inputs_n = torch.zzeros
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
    ctc_loss = torch.nn.CTCLoss()
    for images, targets, _ in metric_logger.log_every(data_loader, print_freq, header):
        count += 1
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # model.eval()
        # detections = model(images)

        # inputs_n = transform(detections)
        # targets_n = transform(targets)

        # target_lengths = torch.tensor([len(targets[i]['labels']) for i in range(len(targets)) ])
        # targets_n = torch.ones((len(targets), max(target_lengths)))
        # # inputs_lengths = torch.tensor([  min(len(detections[i]['labels']), target_lengths[i]) for i in range(len(targets))     ])
        # inputs_lengths = torch.tensor([ len(detections[i]['labels']) for i in range(len(targets))     ])
        # inputs_n = torch.ones(( max(inputs_lengths), len(targets), 91))
        # for i in range(len(target_lengths)):
        #     sort_val, _ = targets[i]['labels'].sort()
        #     targets_n[i, :target_lengths[i]] = sort_val
        #     sort_val, sort_idx = detections[i]['labels'].sort()
        #     # breakpoint()
        #     inputs_n[:inputs_lengths[i], i,  :] = detections[i]['scores_orig_shape'][sort_idx[:inputs_lengths[i]], :]
        # breakpoint()
        # losses += ctc_loss(torch.nn.functional.log_softmax(inputs_n, dim=0), targets_n, inputs_lengths, target_lengths)
        # 
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
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(name)
        #         print(param.grad.sum())
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if args.output_dir and count % 500 == 0:
            model_without_ddp = model
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                # 'optimizer': optimizer.state_dict(),
                # 'lr_scheduler': lr_scheduler.state_dict(),
                'args': args,
                'epoch': epoch},
                os.path.join(args.output_dir, f'model_{epoch}_{count}.pth'))
          
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
    print("Averaged stats:", metric_logger)
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

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))
def find_gt_wl(label, mapping_dict):
    
    for key in mapping_dict:
        if  label in mapping_dict[key]:
            return key
    
    return 'other'
# @torch.no_grad()
# def evaluate_sw(args, model, dataset, device, root):
#     model.eval()
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     header = 'Test:'
#     cpu_device = torch.device("cpu")
#     coco = get_coco_api_from_dataset(dataset)
#     iou_types = _get_iou_types(model)
#     catIds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, \
#          18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,\
#               35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, \
#                   49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, \
#                        61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, \
#                            76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
#     coco_evaluator = CocoEvaluator(coco, iou_types)
#     coco_evaluator.coco_eval['bbox'].params.useCats = 0
#     mapping_dict = read_whitelist(args)
#     priority_list = list(mapping_dict.keys())
#     result_dict_bbox_ag = {'fp': [0 for _ in range(len(priority_list) )], 'fn': [0 for _ in range(len(priority_list) )], 'tp': [0 for _ in range(len(priority_list) )]}

#     result_dict = {'fp': [0 for _ in range(len(priority_list) )], 'fn': [0 for _ in range(len(priority_list) )], 'tp': [0 for _ in range(len(priority_list) )]}
#     count = 0
#     print("lenght of data: ", len(dataset))
    
    
#     acc = 0
#     avg_acc = []
#     all_wl_count = 0 
#     all_others_count = 0
#     acc_dict = {}
#     images_count = {}
#     for key in mapping_dict:
#         acc_dict[key] = []
#     o_count = 0
#     # available_ids = list(pd.read_csv("avail_split7_data.csv")['image_id'])
#     f1_list, recall_list, prec_list, acc_list = [], [], [], []
#     tp, fp, fn = 0, 0, 0
#     for (img, anno, image_id) in dataset:
#         # if image_id not in available_ids:
#         #     continue
#         count += 1
#         o_count = 0
#         acc = 0
        
#         if count % 50 == 0:
#             avg_re = np.mean(np.array(recall_list))
#             avg_prec = np.mean(np.array(prec_list))
#             avg_f1 = 2 * (avg_prec * avg_re) / (avg_prec + avg_re)
#             print(f"Average prec: {avg_prec} average recall: {avg_re} f1: {np.mean(np.array(f1_list))} avg f1: {avg_f1}")
            
#             # print(o_f1)
#         path = coco.loadImgs(image_id)[0]["file_name"]
#         image_raw = cv2.imread(os.path.join(root, path))
#         result, output, top_predictions = predict(image_raw, model, device, mapping_dict)
#         cv2.imwrite(f'outputs/pred_{image_id}.jpg', result)
#         outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in output]
#         res = {int(image_id): outputs[0]}
#         coco_evaluator.update(res)
#         ious = coco_evaluator.coco_eval['bbox'].computeIoU(image_id, 0)
#         pred_wls = []
#         gt_wls =  []
#         pred_labels = output[0]['labels']
#         pred_scores = output[0]['scores']
#         try:
#             for cls_ind in range(len(anno['labels'])):

#                 gt_wl = find_gt_wl(anno['labels'][cls_ind], mapping_dict)
#                 if gt_wl != 'other':
#                     o_count += 1
#                     all_wl_count += 1
                    
                    
#                 else:
#                     all_others_count += 1

#         except:
#             pass
#         for i in range(len(pred_labels)):
#             cls_name = pred_labels[i]
#             if find_gt_wl(cls_name, mapping_dict) != 'other' \
#                 and pred_scores[i] > args.th:
#                 acc += 1
#         tp = min(acc, o_count)
#         fp = max(0, acc - o_count)
#         fn = max(0, o_count - acc)
#         if tp == 0:
#             re = 0
#             prec = 0
#             f1 = 0
#         else:
#             re = tp / (tp + fn)
#             prec = tp / (tp + fp)
#             f1 = 2 * (re * prec) / (re + prec)
#         recall_list.append(re)
#         prec_list.append(prec)
#         f1_list.append(f1)
#         acc_list.append(1 - abs(o_count-acc)/o_count)
#         assert f1 >= 0 and f1 <= 1





@torch.no_grad()
def evaluate_sw(args, model, dataset, device, root):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    cpu_device = torch.device("cpu")
    coco = get_coco_api_from_dataset(dataset)
    iou_types = _get_iou_types(model)
    catIds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, \
         18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,\
              35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, \
                  49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, \
                       61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, \
                           76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    coco_evaluator = CocoEvaluator(coco, iou_types)
    coco_evaluator.coco_eval['bbox'].params.useCats = 0
    mapping_dict = read_whitelist(args)
    priority_list = list(mapping_dict.keys())
    result_dict_bbox_ag = {'fp': [0 for _ in range(len(priority_list) )], 'fn': [0 for _ in range(len(priority_list) )], 'tp': [0 for _ in range(len(priority_list) )]}

    result_dict = {'fp': [0 for _ in range(len(priority_list) )], 'fn': [0 for _ in range(len(priority_list) )], 'tp': [0 for _ in range(len(priority_list) )]}
    count = 0
    print("lenght of data: ", len(dataset))
    
    
    acc = 0
    avg_acc = []
    all_wl_count = 0 
    all_others_count = 0
    acc_dict = {}
    images_count = {}
    for key in mapping_dict:
        acc_dict[key] = []
    o_count = 0
    # available_ids = list(pd.read_csv("avail_split7_data.csv")['image_id'])
    f1_list, recall_list, prec_list, acc_list = [], [], [], []
    tp, fp, fn = 0, 0, 0
    for (img, anno, image_id) in dataset:
        # if image_id not in available_ids:
        #     continue
        count += 1
        o_count = 0
        acc = 0
        
        if count % 50 == 0:
            avg_re = np.mean(np.array(recall_list))
            avg_prec = np.mean(np.array(prec_list))
            avg_f1 = 2 * (avg_prec * avg_re) / (avg_prec + avg_re)
            print(f"Average prec: {avg_prec} average recall: {avg_re} f1: {np.mean(np.array(f1_list))} avg f1: {avg_f1}")
            
            # print(o_f1)
        path = coco.loadImgs(image_id)[0]["file_name"]
        image_raw = cv2.imread(os.path.join(root, path))
        # result, output, top_predictions = predict(image_raw, model, device, mapping_dict)
        result = image_raw.copy()
        img = image_raw[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3 x 416 x 416
        img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        img = torch.from_numpy(img)

        with torch.no_grad():
            output = model([img.to(device)])
        # cv2.imwrite(f'outputs/pred_{image_id}.jpg', result)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in output]
        res = {int(image_id): outputs[0]}
        coco_evaluator.update(res)
        ious = coco_evaluator.coco_eval['bbox'].computeIoU(image_id, 0)
        pred_wls = []
        gt_wls =  []

        pred_labels = output[0]['labels']
        pred_scores = output[0]['scores']
        our_scores = output[0]['scores_count']
        for cls_ind in range(len(anno['labels'])):

            gt_wl = find_gt_wl(anno['labels'][cls_ind], mapping_dict)
            if gt_wl != 'other':
                o_count += 1
                all_wl_count += 1
                
                
            else:
                all_others_count += 1

        for i in range(len(pred_labels)):
            cls_name = pred_labels[i]
            if find_gt_wl(cls_name, mapping_dict) != 'other' \
                and pred_scores[i] > args.th:
                acc += 1
        acc = torch.argmax(our_scores).item()
        print(f"acc: {acc} o_count: {o_count}")
        tp = min(acc, o_count)
        fp = max(0, acc - o_count)
        fn = max(0, o_count - acc)
        if tp == 0:
            re = 0
            prec = 0
            f1 = 0
        else:
            re = tp / (tp + fn)
            prec = tp / (tp + fp)
            f1 = 2 * (re * prec) / (re + prec)
        recall_list.append(re)
        prec_list.append(prec)
        f1_list.append(f1)
        acc_list.append(1 - abs(o_count-acc)/o_count)
        assert f1 >= 0 and f1 <= 1