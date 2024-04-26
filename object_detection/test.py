import datetime
import os
import time

import torch
import torch.utils.data
import torchvision
import models
import pandas as pd
import os
from dataset.coco_utils import get_coco, get_coco_kp

from group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from engine import train_one_epoch, evaluate, evaluate_sw, read_whitelist, evaluate_sw_old

import utils
import transforms as T


def get_dataset(args,split,  name, image_set, transform, data_path, extra_input=None):
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


def main(args):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # Data loading code

    dataset_test, num_classes = get_dataset(args, args.split, args.dataset, "val", get_transform(train=False), args.data_path, args.extra_input)

    if args.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        test_sampler = torch.utils.data.RandomSampler(dataset_test)


    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    mapping_dict = read_whitelist(args)
    # model =torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    if os.environ["WL_EVAL"] == "True" :
        
        model = models.__dict__[args.model](num_classes=len(mapping_dict["wl1"]) + 1,
                                            pretrained=args.pretrained, 
                                            mapping_dict=mapping_dict,
                                            our_loss=args.our_loss)
    elif os.environ["NEW"] == "True":
        model = models.__dict__[args.model](num_classes=int(os.environ['CLASSES']),
                                            pretrained=args.pretrained, 
                                            mapping_dict=mapping_dict,
                                            our_loss=args.our_loss)
    else:
        model = models.__dict__[args.model](num_classes=91, pretrained=args.pretrained, our_loss=args.our_loss)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
    if args.resume:
        state = torch.load(args.resume, map_location='cpu')
        pretrained_dict = {k: v for k, v in state['model'].items() if k in model.state_dict()}
        model.load_state_dict(pretrained_dict, strict=True)

    
    start_time = time.time()
    
        # evaluate after every epoch
    f1 = evaluate_sw_old(args, model, dataset_test, device=device, root=os.path.join(args.data_path, 'val2014'))

    if f"results_{args.method}.csv" not in os.listdir("results"):
        df = pd.DataFrame({"app": [args.app], "precision": [f1]})
        df.to_csv(f"results/results_{args.method}.csv", index=False)
    else:
        df = pd.read_csv(f"results/results_{args.method}.csv")
        df2 = pd.DataFrame({"app": [args.app], "precision": [f1]})
        df = pd.concat([df, df2], ignore_index=True)
        df.to_csv(f"results/results_{args.method}.csv", index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--app', type=str,)
    parser.add_argument('--app_type', type=str,)
    parser.add_argument('--method', type=str,)
    parser.add_argument('--data-path', default='/dataheart/yuhanl.used/coco/',
                        help='dataset')
    parser.add_argument('--dataset', default='coco',
                        help='dataset')
    parser.add_argument('--model', default='maskrcnn_resnet50_fpn',
                        help='model')
    parser.add_argument('--device', default='cuda',
                        help='device')
    parser.add_argument('--batch-size', default=1, type=int,
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
    parser.add_argument('--wl_path_test', type=str,
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
    parser.add_argument('--our_loss', default=False, help='if distribute or not')

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--iou_th', default=0.3, type=float,
                        help='start epoch')
    parser.add_argument('--th', default=0.3, type=float,
                        help='start epoch')
    args = parser.parse_args()

    if args.output_dir:
        utils.mkdir(args.output_dir)

    main(args)
