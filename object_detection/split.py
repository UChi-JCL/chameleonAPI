r"""PyTorch Detection Training.

To run in a multi-gpu environment, use the distributed launcher::

    python -m torch.distributed.launch --nproc_per_node=$NGPU --use_env \
        train.py ... --world-size $NGPU

The default hyperparameters are tuned for training on 8 gpus and 2 images per gpu.
    --lr 0.02 --batch-size 2 --world-size 8
If you use different number of gpus, the learning rate should be changed to 0.02/8*$NGPU.

On top of that, for training Faster/Mask R-CNN, the default hyperparameters are
    --epochs 26 --lr-steps 16 22 --aspect-ratio-group-factor 3

Also, if you train Keypoint R-CNN, the default hyperparameters are
    --epochs 46 --lr-steps 36 43 --aspect-ratio-group-factor 3
Because the number of images is smaller in the person keypoint subset of COCO,
the number of epochs should be adapted so that we have the same number of iterations.
"""
import datetime
import os
import time

import torch
import torch.utils.data
import torchvision
import models
import random
from dataset.coco_utils import get_coco, get_coco_kp

from group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from engine import train_one_epoch, evaluate, evaluate_sw, read_whitelist

import utils
import transforms as T


def get_dataset(args, split,  name, image_set, transform, data_path, extra_input=None):
    paths = {
        "coco": (data_path, get_coco, 91),
        "coco_kp": (data_path, get_coco_kp, 2)
    }
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(args, split, p, image_set=image_set, transforms=transform, extra_input=extra_input)
    return ds, num_classes


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def intersection(list1, list2): 
    return list(set(list1).intersection(set(list2)))
def main(args):
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    # Data loading code
    print("Loading data")

    dataset_test, num_classes = get_dataset(args, args.split, args.dataset, args.split, get_transform(train=False), args.data_path)

    mapping_dict = read_whitelist(args)
    training_stats = {}
    for key in mapping_dict:
        training_stats[key] = 0
    image_stats = {}
    image_ids = {}
    for key in mapping_dict:
        image_stats[key] = 0
        image_ids[key] = []
    priority_list = list(mapping_dict.keys())
    cnt = 0
    for (img, anno, image_id) in dataset_test:
        cnt += 1
        if cnt % 1000 == 0:
            print(cnt)
        # if cnt > 5000: break
        flag_wl = False
        for key in mapping_dict:
            if key == "other": continue
            if len(intersection(anno["labels"].numpy(), mapping_dict[key])) > 0:
                image_stats[key] += 1
                image_ids[key].append(image_id)
                flag_wl = True
        if flag_wl == False:
            image_stats["other"] += 1
            image_ids["other"].append(image_id)   
    df = {"image_id": [], "wl": [], "split": []} 
    for key in mapping_dict:
        random.seed(0)
        random.shuffle(image_ids[key])
        if key == "other": 
            train_size = int(args.limit * 0.8)
            test_size = int(args.limit * 0.2)
        else:
            train_size = int(len(image_ids[key]) * 0.8)
            test_size = int(len(image_ids[key]) * 0.2)
        df['image_id'] += image_ids[key][:train_size]
        df['wl'] += [key] * len( image_ids[key][:train_size])
        df['split'] += ['train'] * len( image_ids[key][:train_size])
        print(key, len(image_ids[key][:train_size]))        
        df['image_id'] += image_ids[key][train_size:train_size+test_size]
        df['wl'] += [key] * len( image_ids[key][train_size:train_size+test_size])
        df['split'] += ['test'] * len( image_ids[key][train_size:train_size+test_size])
    import pandas as pd
    
    df = pd.DataFrame(df)
    df.to_csv(args.output, index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--data-path', default='/dataheart/yuhanl.used/coco/',
                        help='dataset')
    parser.add_argument('--dataset', default='coco',
                        help='dataset')
    parser.add_argument('--model', default='maskrcnn_resnet50_fpn',
                        help='model')
    parser.add_argument('--device', default='cuda',
                        help='device')
    parser.add_argument('--batch-size', default=2, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('--epochs', default=26, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--lr', default=0.02, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                        'on 8 gpus and 2 images_per_gpu')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--lr-step-size', default=8, type=int,
                        help='decrease lr every step-size epochs')
    parser.add_argument('--lr-steps', default=[16, 22], nargs='+', type=int,
                        help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float,
                        help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=20, type=int,
                        help='print frequency')
    parser.add_argument('--extra_input', type=str,
                        help='print frequency')
    parser.add_argument('--split', type=str,
                        help='print frequency')
    parser.add_argument('--wl_path', type=str,
                        help='print frequency')
    parser.add_argument('--output-dir', default='checkpoints',
                        help='path where to save')
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='start epoch')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    parser.add_argument("--test-only", action="store_true",
                        help="Only test the model")
    parser.add_argument("--pretrained", action="store_true",
                        help="Use pre-trained models from the modelzoo")

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--iou_th', default=0.3, type=float,
                        help='start epoch')
    parser.add_argument('--th', default=0.3, type=float,
                        help='start epoch')
    parser.add_argument('--limit', default=1000, type=int,
                        help='start epoch')
    parser.add_argument('--output', default='test.csv', type=str)
    args = parser.parse_args()

    if args.output_dir:
        utils.mkdir(args.output_dir)

    main(args)
