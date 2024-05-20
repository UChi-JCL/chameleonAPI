import os
from copy import deepcopy
import random
import time
from copy import deepcopy

import numpy as np
from PIL import Image
from torchvision import datasets as datasets
import torch
from PIL import ImageDraw
from pycocotools.coco import COCO
import pandas as pd

def parse_args(parser):
    # parsing args
    args = parser.parse_args()
    if args.dataset_type == 'OpenImages':
        args.do_bottleneck_head = True
        if args.th == None:
            args.th = 0.995
    else:
        args.do_bottleneck_head = False
        if args.th == None:
            args.th = 0.7
    return args


def average_precision(output, target):
    epsilon = 1e-8
    # sort examples
    indices = output.argsort()[::-1]
    # Computes prec@i
    total_count_ = np.cumsum(np.ones((len(output), 1)))

    target_ = target[indices]
    ind = target_ == 1
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_ / (total + epsilon)

    return precision_at_i


def mAP(targs, preds):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """
    testing_classes = pd.read_csv("testing_classes.csv")
    all_index = list(testing_classes['class_index'])
    all_index = np.array(all_index)
    # preds = preds[:, all_index]
    # targs = targs[:, all_index]
    if np.size(preds) == 0:
        return 0
    ap = np.zeros((preds.shape[1]))
    # compute average precision for each class
    for k in range(preds.shape[1]):
        # sort scores
        scores = preds[:, k]
        targets = targs[:, k]
        # compute average precision
        ap[k] = average_precision(scores, targets)
    return 100 * ap.mean()

def mAP_whitelist(targs, preds, W):
    """Returns the model's average precision for each class, preds shape is still of original shape, but we only calculate the values in the whitelist
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """

    if np.size(preds) == 0:
        return 0
    ap = np.zeros((len(W)))
    # compute average precision for each class
    for k in range(len(W)):
        # sort scores
        cls_idx = W[k]
        scores = preds[:, cls_idx]
        targets = targs[:, cls_idx]
        # compute average precision
        ap[k] = average_precision(scores, targets)
    return 100 * ap.mean()
def mAP_whitelist_sub(targs, preds, W):
    """Returns the model's average precision for each class, when preds is of shape len(W)
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """
    Sig = torch.nn.Sigmoid()
    if np.size(preds) == 0:
        return 0
    ap = np.zeros((len(W)))
    # compute average precision for each class
    th = 0.8
    for k in range(len(W)):
        # sort scores
        # cls_idx = W[k]
        # scores = preds[:, cls_idx]
        # targets = targs[:, cls_idx]
        scores = preds[:, k]
        targets = targs[:, k]
        # print(scores)
        ap[k] += (len(scores[scores>th])/len(scores))
        # compute average precision
    return 100 * ap.mean()
class AverageMeter(object):
    def __init__(self):
        self.val = None
        self.sum = None
        self.cnt = None
        self.avg = None
        self.ema = None
        self.initialized = False

    def update(self, val, n=1):
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def initialize(self, val, n):
        self.val = val
        self.sum = val * n
        self.cnt = n
        self.avg = val
        self.ema = val
        self.initialized = True

    def add(self, val, n):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
        self.ema = self.ema * 0.99 + self.val * 0.01


class OpenImage:
    def __init__(self, root, dataset_file, transform=None, target_transform=None, start_idx=0, end_idx=0):
        self.data_file = pd.read_csv(dataset_file)[start_idx:end_idx]
        self.root = root
        self.ids = []
        for i in range(len(self.data_file)):
            if self.data_file.iloc[i]['split'] == 'train':
                self.ids.append(self.data_file.iloc[i]['image_id'])
        self.transform = transform
        self.target_transform = target_transform
        self.mid_to_human_class_file = pd.read_csv("oidv6-class-descriptions.csv")
        self.class_list_file = pd.read_csv("all_classes.csv")
        state = torch.load("Open_ImagesV6_TRresNet_L_448.pth", map_location='cpu')

        self.class_list = list(state['idx_to_class'].values())
        self.class_list = [i.replace("'", "") for i in self.class_list]

        self.mid_to_human_class = {}
        self.start_idx = start_idx
        self.end_idx = end_idx
        for i in range(len(self.mid_to_human_class_file)):
            self.mid_to_human_class[self.mid_to_human_class_file.iloc[i][0]] = self.mid_to_human_class_file.iloc[i][1]


    def __getitem__(self, index):

        img_id = self.ids[index]
        index = list(self.data_file['image_id']).index(img_id)

        output = torch.zeros(9605, dtype=torch.long)

        img = Image.open(os.path.join(self.root, img_id)).convert('RGB')
        all_classes = self.data_file.iloc[index]['class_list']
        all_classes = all_classes.split('[')[1].split(']')[0].replace("'", "").split(', ')

        for class_name in all_classes:
            if self.mid_to_human_class[class_name] in self.class_list:
                output[self.class_list.index(self.mid_to_human_class[class_name])] = 1
        if self.transform is not None:
            img = self.transform(img)
        target = output
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
    def __len__(self):
        return len(self.ids)

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))


def define_whitelist_wl(all_classes, mapping_dict):
    
    
    existing_classes = []
    for class_name in mapping_dict:
        if len(intersection(mapping_dict[class_name], all_classes)) > 0:
            existing_classes.append(class_name)
    # print(all_classes)
    # print("WL gt", existing_classes)

    return list(set(existing_classes))
class OpenImageW:
    def __init__(self, root, dataset_file, wl_path=None, transform=None, target_transform=None, start_idx=0, end_idx=0):
        self.data_file = pd.read_csv(dataset_file)[start_idx:end_idx]
        self.root = root
        self.ids = []
        
        self.transform = transform
        self.target_transform = target_transform
        self.mid_to_human_class_file = pd.read_csv("oidv6-class-descriptions.csv")
        self.class_list_file = pd.read_csv("all_classes.csv")
        state = torch.load("Open_ImagesV6_TRresNet_L_448.pth", map_location='cpu')

        self.class_list = list(state['idx_to_class'].values())
        self.class_list = [i.replace("'", "") for i in self.class_list]

        self.mid_to_human_class = {}
        self.start_idx = start_idx
        self.end_idx = end_idx
        for i in range(len(self.mid_to_human_class_file)):
            self.mid_to_human_class[self.mid_to_human_class_file.iloc[i][0]] = self.mid_to_human_class_file.iloc[i][1]
        whitelist_mapping = {}
        if wl_path is not None:
            wl_mapping = pd.read_csv(wl_path)
            all_classes_count = {}

            for i in range(len(wl_mapping)):
                if wl_mapping.iloc[i]['wl'] not in whitelist_mapping:
                    whitelist_mapping[wl_mapping.iloc[i]['wl']] = []
                    all_classes_count[wl_mapping.iloc[i]['wl']] = 0
                whitelist_mapping[wl_mapping.iloc[i]['wl']].append(wl_mapping.iloc[i]['class_name'])
            self.whitelist_mapping = whitelist_mapping
            all_whitelist_classes = []
            for key in self.whitelist_mapping:
                all_whitelist_classes.extend(self.whitelist_mapping[key])
        
        for i in range(len(self.data_file)):
            all_classes = self.data_file.iloc[i]['class_list']
            all_classes = all_classes.split('[')[1].split(']')[0].replace("'", "").split(', ')
            all_classes_human_read = [self.mid_to_human_class[i] for i in all_classes]
            all_classes_neg = self.data_file.iloc[i]['class_list_neg']
            all_classes_neg = all_classes_neg.split('[')[1].split(']')[0].replace("'", "").split(', ')
            all_neg_classes_human_read =  []
            if all_classes_neg[0] != '':
                all_neg_classes_human_read = [self.mid_to_human_class[i] for i in all_classes_neg]
            # if len(intersection(all_classes_human_read, all_whitelist_classes)) == 0 and len(intersection(all_neg_classes_human_read, all_whitelist_classes)) == 0:
            #     continue
            gt_classes = define_whitelist_wl(all_classes_human_read, self.whitelist_mapping)
            gt_neg_classes = define_whitelist_wl(all_neg_classes_human_read,  self.whitelist_mapping)
            # if len(intersection(gt_classes, gt_neg_classes)) > 0: continue
            if self.data_file.iloc[i]['split'] == 'train':
                self.ids.append(self.data_file.iloc[i]['image_id'])

    def __getitem__(self, index):

        img_id = self.ids[index]
        index = list(self.data_file['image_id']).index(img_id)

        output = torch.zeros(9605, dtype=torch.long)

        img = Image.open(os.path.join(self.root, img_id)).convert('RGB')
        all_classes = self.data_file.iloc[index]['class_list']
        all_classes = all_classes.split('[')[1].split(']')[0].replace("'", "").split(', ')

        all_classes_neg = self.data_file.iloc[index]['class_list_neg']
        all_classes_neg = all_classes_neg.split('[')[1].split(']')[0].replace("'", "").split(', ')
        output_neg = torch.zeros(9605, dtype=torch.long)
        for class_name in all_classes:
            if self.mid_to_human_class[class_name] in self.class_list:
                output[self.class_list.index(self.mid_to_human_class[class_name])] = 1
    
        for class_name in all_classes_neg:
            if class_name != '' and self.mid_to_human_class[class_name] in self.class_list:
                output_neg[self.class_list.index(self.mid_to_human_class[class_name])] = 1
        if self.transform is not None:
            img = self.transform(img)
        target = output
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, output_neg
    def __len__(self):
        return len(self.ids)



class CocoDetection(datasets.coco.CocoDetection):
    def __init__(self, root, annFile, transform=None, target_transform=None):
        self.root = root
        self.coco = COCO(annFile)

        self.ids = list(self.coco.imgToAnns.keys())
        self.transform = transform
        self.target_transform = target_transform
        self.cat2cat = dict()
        for cat in self.coco.cats.keys():
            self.cat2cat[cat] = len(self.cat2cat)
        # print(self.cat2cat)

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        output = torch.zeros((3, 80), dtype=torch.long)
        for obj in target:
            if obj['area'] < 32 * 32:
                output[0][self.cat2cat[obj['category_id']]] = 1
            elif obj['area'] < 96 * 96:
                output[1][self.cat2cat[obj['category_id']]] = 1
            else:
                output[2][self.cat2cat[obj['category_id']]] = 1
        target = output

        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


class CutoutPIL(object):
    def __init__(self, cutout_factor=0.5):
        self.cutout_factor = cutout_factor

    def __call__(self, x):
        img_draw = ImageDraw.Draw(x)
        h, w = x.size[0], x.size[1]  # HWC
        h_cutout = int(self.cutout_factor * h + 0.5)
        w_cutout = int(self.cutout_factor * w + 0.5)
        y_c = np.random.randint(h)
        x_c = np.random.randint(w)

        y1 = np.clip(y_c - h_cutout // 2, 0, h)
        y2 = np.clip(y_c + h_cutout // 2, 0, h)
        x1 = np.clip(x_c - w_cutout // 2, 0, w)
        x2 = np.clip(x_c + w_cutout // 2, 0, w)
        fill_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img_draw.rectangle([x1, y1, x2, y2], fill=fill_color)

        return x


def add_weight_decay(model, weight_decay=1e-4, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]
