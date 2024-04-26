from re import L
import utils
import dataset.transforms as T
import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn
from dataset.coco_utils import get_coco, get_coco_kp
from engine import train_one_epoch, evaluate, read_whitelist
from dataset.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
import argparse
import torchvision
import models
import cv2
import random

def get_args():
    parser = argparse.ArgumentParser(description='Pytorch Faster-rcnn Training')

    parser.add_argument('--data_path', default='/dataheart/yuhanl.used/coco/', help='dataset path')
    parser.add_argument('--model', default='fasterrcnn_resnet50_fpn', help='model')
    parser.add_argument('--dataset', default='coco', help='dataset')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--b', '--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=3, type=int, metavar='N',
                        help='number of total epochs to run')    
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.02, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    parser.add_argument('--lr-step-size', default=8, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-steps', default=[8, 11], nargs='+', type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--test_only', default=False, type=bool, help='resume from checkpoint')
    parser.add_argument('--output-dir', default='./result', help='path where to save')
    parser.add_argument('--aspect-ratio-group-factor', default=0, type=int)
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )
    parser.add_argument('--distributed', default=True, help='if distribute or not')
    parser.add_argument('--parallel', default=False, help='if distribute or not')
    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--split', default='test', help='dataset')
    parser.add_argument('--wl_path', default='test', help='dataset')
    parser.add_argument('--extra_input', default='test', help='dataset')
    parser.add_argument('--output_dir', default=None, type=str, help='dataset')
    parser.add_argument('--our_loss', default=False, help='if distribute or not')

    args = parser.parse_args()

    return args


def get_dataset(args,split,  name, image_set, transform, data_path, extra_input=None):
    paths = {
        "coco": ('/dataheart/yuhanl.used/coco/', get_coco, 91),
        "coco_kp": ('/datasets01/COCO/022719/', get_coco_kp, 2)
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


def main():
    args = get_args()
    if args.output_dir:
        utils.mkdir(args.output_dir)    
    utils.init_distributed_mode(args)
    mapping_dict = read_whitelist(args)
    model = models.__dict__[args.model](num_classes=3,
                                        pretrained=args.pretrained, 
                                        mapping_dict=mapping_dict,
                                        our_loss=args.our_loss)
    # for name, param in model.named_parameters():
    #     if 'backbone.body' in name:
    #         layer_no = int(name.split("backbone.body.")[1].split(".")[0])
    #         if layer_no < 2:
    #             param.requires_grad = False
    # Data loading
    print("Loading data")
    dataset, num_classes = get_dataset(args, args.split, args.dataset, "train", get_transform(train=True), args.data_path, args.extra_input)
    dataset_test, num_classes = get_dataset(args, 'test', args.dataset, "train", get_transform(train=False), args.data_path, args.extra_input)
    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    if args.aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.b)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, args.b, drop_last=True)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.b,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    # Model creating
    print("Creating model")
    
    device = torch.device(args.device)
    model.to(device)
    # Distribute
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module    

    # Parallel
    if args.parallel:
        print('Training parallel')
        model = torch.nn.DataParallel(model).cuda()
        model_without_ddp = model.module

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

    # Resume training
    if args.resume:
        print('Resume training')
        state = torch.load(args.resume, map_location='cpu')
        pretrained_dict = {k: v for k, v in state['model'].items() if k in model.state_dict()}
        model.load_state_dict(pretrained_dict, strict=False)

        optimizer.load_state_dict(state['optimizer'])
        lr_scheduler.load_state_dict(state['lr_scheduler'])

    if args.test_only:
        evaluate(model, data_loader_test, device=device)
        return

    # Training
    print('Start training')
    start_time = time.time()
    for epoch in range(args.epochs):
        train_one_epoch(args, model, optimizer, data_loader, device, epoch, args.print_freq)
        lr_scheduler.step()
        if args.output_dir:
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'args': args},
                os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))

        # evaluate after every epoch
        evaluate(model, data_loader_test, device=device)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
      

if __name__ == "__main__":
    main()