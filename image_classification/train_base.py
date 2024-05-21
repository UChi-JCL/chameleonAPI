from builtins import breakpoint
import os
import argparse
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from src.helper_functions.helper_functions import mAP, CocoDetection, CutoutPIL, ModelEma, add_weight_decay, OpenImageW, mAP_whitelist, mAP_whitelist_sub, OpenImage
from src.models import create_model
from src.loss_functions.losses import  AsymmetricLossOrig, AsymmetricLossOrigNew
from randaugment import RandAugment
from torch.cuda.amp import GradScaler, autocast
import pandas as pd
from torch import nn
import time
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
# torch.multiprocessing.set_start_method('spawn')
class bottleneck_head(nn.Module):
    def __init__(self, num_features, num_classes, bottleneck_features=200):
        super(bottleneck_head, self).__init__()
        self.embedding_generator = nn.ModuleList()
        self.embedding_generator.append(nn.Linear(num_features, bottleneck_features))
        self.embedding_generator = nn.Sequential(*self.embedding_generator)
        self.FC = nn.Linear(bottleneck_features, num_classes)

    def forward(self, x):
        self.embedding = self.embedding_generator(x)
        logits = self.FC(self.embedding)
        return logits

parser = argparse.ArgumentParser(description='PyTorch MS_COCO Training')
parser.add_argument('data', metavar='DIR', help='path to dataset', default='subset_training.csv')
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--epochs', default=10, type=int)

parser.add_argument('--model_name', default='tresnet_l')
parser.add_argument('--model_path', default=None, type=str)
parser.add_argument('--num-classes', default=9605)
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--image-size', default=448, type=int,
                    metavar='N', help='input image size (default: 448)')
parser.add_argument('--thre', default=0.8, type=float,
                    metavar='N', help='threshold value')
parser.add_argument('-b', '--batch-size', default=12, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--print-freq', '-p', default=8, type=int,
                    metavar='N', help='print frequency (default: 64)')
parser.add_argument('--root', default='/tank/yuhanl/openimages/', type=str, help='print frequency (default: 64)')
parser.add_argument('--dataset_type', type=str, default='OpenImages')
parser.add_argument('--ckpt_path', type=str, default='OpenImages')

parser.add_argument('--input_size', type=int, default=448)
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--rank', type=int, default=0)
parser.add_argument('--world-size', type=int, default=1)
parser.add_argument('--backend', type=str, default='gloo')
parser.add_argument('--ckpt_step', type=int, default=0)

parser.add_argument('--resume', type=str, default=None)

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--limit', type=int, default=15)
parser.add_argument('--gamma_neg', type=int, default=4)
parser.add_argument('--gamma_pos', type=int, default=4)

parser.add_argument('--alpha', type=int, default=7)
parser.add_argument('--alpha1', type=float, default=7)
parser.add_argument('--alpha_other', type=int, default=7)
parser.add_argument('--alpha3', type=float, default=0.5)
parser.add_argument('--stop_epoch', type=int, default=5)

parser.add_argument('--alpha2', type=int, default=2)
parser.add_argument('--focal', action="store_true")
parser.add_argument('--large', action="store_true")
parser.add_argument('--rankloss', action="store_true")
parser.add_argument('--ranklossnew', action="store_true")
parser.add_argument('--asymm', action="store_true")
parser.add_argument('--priority', action="store_true")
parser.add_argument('--penalize_other', action="store_true")
parser.add_argument('--optimize', action="store_true")
parser.add_argument('--final', action="store_true")
parser.add_argument('--small', action="store_true")
parser.add_argument('--small_test', action="store_true")
parser.add_argument('--frozen', action="store_true")
parser.add_argument('--sigmoid', action="store_true")
parser.add_argument('--frozen_all', action="store_true")
parser.add_argument('--cycle', action="store_true")
parser.add_argument('--binary', action="store_true")

parser.add_argument('--weight_balancing', action="store_true")
parser.add_argument('--model_path_openimages', type=str, default='/dataheart/yuhanl/Open_ImagesV6_TRresNet_L_448.pth')
parser.add_argument("--wl_path", type=str, default='data/sub_training.csv')


args = parser.parse_args()


def main():
    args.do_bottleneck_head = True
    # mapping_dict = { 'clothing': ['Clothing'], 'shirt': ['Shirt'], 'pants': ['Pantsuit'], 'jacket': ['Jacket'], 'footwear': ['Footwear'], 'shoe': ['Shoe'], 'paper': [ 'Paper bag', 'Paper product', 'Paper towel', 'Paper'], 'glass': ['Glass'], 'carton': ['Carton'], 'cardboard': ['Cardboard'], 'tin': ['Tin can', 'Tin'], 'metal': [ 'Metal'], 'plastic': ['Plastic arts', 'Plastic bag', 'Plastic bottle', 'Plastic wrap', 'Plastic']}
    mapping_dict = {'food': ['Food','Food grain', 'Food group',  'Food storage containers', 'Food storage', ], 'snack': ['Fruit snack', 'Snack cake', 'Snack'], 'compost': ['Compost'], 'clothing': ['Clothing'], 'shirt': ['Shirt'], 'pants': ['Pantsuit'], 'jacket': ['Jacket'], 'footwear': ['Footwear'], 'shoe': ['Shoe'], 'paper': [ 'Paper bag', 'Paper product', 'Paper towel', 'Paper'], 'glass': ['Glass'], 'carton': ['Carton'], 'cardboard': ['Cardboard'], 'tin': ['Tin can', 'Tin'], 'metal': [ 'Metal'], 'plastic': ['Plastic arts', 'Plastic bag', 'Plastic bottle', 'Plastic wrap', 'Plastic']}
    # # mapping_dict = {'paper': ['art paper', 'construction paper', 'household paper product', 'origami paper', 'paper bag', 'paper lantern', 'paper product', 'paper towel', 'paper', 'photographic paper', 'rice paper', 'tissue paper', 'toilet paper', 'wrapping paper'], 'glass': ['beer glass', 'glass bottle', 'glass', 'highball glass', 'magnifying glass', 'martini glass', 'old fashioned glass', 'pint glass', 'shot glass', 'stained glass', 'wine glass'], 'carton': ['carton'], 'cardboard': ['cardboard'], 'tin': ['tin can', 'tin'], 'metal': ['foil (metal)', 'metal', 'metallophone', 'metalsmith', 'metalworking hand tool', 'metalworking'], 'plastic': ['plastic arts', 'plastic bag', 'plastic bottle', 'plastic wrap', 'plastic'], }

    class_list_file = pd.read_csv("data/all_classes.csv")
    state = torch.load(args.model_path_openimages, map_location='cpu')

    class_list = list(state['idx_to_class'].values())
    class_list = [i.replace("'", "").replace("\"", "") for i in class_list]
    W = []
    W_human = []
    for key in mapping_dict.keys():
        for value in mapping_dict[key][:args.limit]:
            if value in class_list:
                W.append(class_list.index(value))
                W_human.append(value)
    # Setup model


    args.do_bottleneck_head = True
    model = create_model(args)

    # for name, param in model.named_parameters():
    #     print(name)
    if args.model_path:  # make sure to load pretrained ImageNet model
        state = torch.load(args.model_path, map_location='cpu')
        model.load_state_dict(state['model'], strict=True)
    


    # model.head.fc = bottleneck_head(model.num_features, len(W)+1)
    if args.frozen:
        for name, param in model.named_parameters():
            if "body" in name:
                param.requires_grad = False
            if "body.layer4" in name:
                param.requires_grad = True
    if args.frozen_all:
        for name, param in model.named_parameters():
            if "body" in name:
                param.requires_grad = False
    wl_mapping = pd.read_csv(args.wl_path)
    all_classes_count = {}
    whitelist_mapping = {}
    for i in range(len(wl_mapping)):
        if wl_mapping.iloc[i]['wl'] not in whitelist_mapping:
            whitelist_mapping[wl_mapping.iloc[i]['wl']] = []
            all_classes_count[wl_mapping.iloc[i]['wl']] = 0
        whitelist_mapping[wl_mapping.iloc[i]['wl']].append(wl_mapping.iloc[i]['class_name'])
    mapping_dict = whitelist_mapping    
    all_size = len(pd.read_csv(args.data))
    val_size =  int(all_size * 0.1)
    whole_dataset = OpenImageW(args.root, args.data,
                               wl_path=args.wl_path,
                                transform=transforms.Compose([
                                    transforms.Resize((args.image_size, args.image_size)),
                                    transforms.ToTensor(),
                                    # normalize, # no need, toTensor does normalization
                                ]), start_idx = 0, end_idx = all_size)
    # train_dataset, val_dataset = torch.utils.data.random_split(whole_dataset, [all_size-val_size, val_size])
    # train_dataset = OpenImageW(args.root, args.data,
    #                               transforms.Compose([
    #                                   transforms.Resize((args.image_size, args.image_size)),
    #                                   CutoutPIL(cutout_factor=0.5),
    #                                   RandAugment(),
    #                                   transforms.ToTensor(),
    #                                   # normalize,
    #                               ]), start_idx = val_size, end_idx = all_size, W=W)
    # val_dataset = OpenImage(args.root, args.data,
    #                             transforms.Compose([
    #                                 transforms.Resize((args.image_size, args.image_size)),
    #                                 transforms.ToTensor(),
    #                                 # normalize, # no need, toTensor does normalization
    #                             ]), start_idx = 0, end_idx = val_size)
    # train_dataset = OpenImage(args.root, args.data,
    #                               transforms.Compose([
    #                                   transforms.Resize((args.image_size, args.image_size)),
    #                                   CutoutPIL(cutout_factor=0.5),
    #                                   RandAugment(),
    #                                   transforms.ToTensor(),
    #                                   # normalize,
    #                               ]), start_idx = val_size, end_idx = all_size)

    torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda', args.local_rank)
    model.to(device)
    print(device)



    # Pytorch Data loader
    train_loader = torch.utils.data.DataLoader(
        whole_dataset, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, shuffle = True)
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=args.batch_size,
    #     num_workers=0, pin_memory=True, sampler=train_sampler)

    # Actuall Training
    train_multi_label_coco(args, model, train_loader, train_loader, args.lr, W, W_human, device, mapping_dict, args.limit, args.alpha)


def train_multi_label_coco(args, model, train_loader, val_loader, lr, W, W_human, device, mapping_dict,limit, alpha):
    ema = ModelEma(model, 0.9997)  # 0.9997^641=0.82

    # set optimizer
    Epochs = args.epochs
    Stop_epoch = args.stop_epoch
    weight_decay = 1e-4
    criterion = AsymmetricLossOrigNew(gamma_neg=args.gamma_neg, gamma_pos=args.gamma_pos, clip=0.05, disable_torch_grad_focal_loss=True)

    parameters = add_weight_decay(model, weight_decay)
    optimizer = torch.optim.Adam(params=parameters, lr=lr, weight_decay=0)  # true wd, filter_bias_and_bn
    steps_per_epoch = len(train_loader)
    # scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=Epochs,
    #                                     pct_start=0.2)
    if args.cycle:
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=Epochs,
                                            pct_start=0.2)
    else:
        scheduler = lr_scheduler.StepLR(optimizer, step_size = 1, gamma=0.1)
    highest_mAP = 0
    trainInfoList = []
    scaler = GradScaler()
    start_time = time.time()
    # model.eval()
    if os.path.exists(args.ckpt_path) is False:
        os.makedirs(args.ckpt_path)
    # mAP_score = validate_multi(val_loader, model, ema, W, device)
    for epoch in range(Epochs):
        if epoch > Stop_epoch:
            break

        for i, (inputData, target, target_neg) in enumerate(train_loader):
            model.train()
            inputData = inputData.to(device)
            # target = target.max(dim=1)[0]
            # print("target: ", target.shape)

            with autocast():  # mixed precision
                output = model(inputData).float()
            target = target.to(device)  # (batch,3,num_classes)
            target_neg = target_neg.to(device)
            loss = criterion(output, target, target_neg)

            # print("Prediction: ", output.shape)
            if args.world_size > 1:
                loss = loss.sum()
            model.zero_grad()
            scaler.scale(loss).backward()
            # loss.backward()
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(torch.sum(param.grad))

            scaler.step(optimizer)
            scaler.update()
            if args.cycle:
                scheduler.step()
            # optimizer.step()


            ema.update(model)
            # store information
            if i % 100 == 0 and args.rank == 0:
                start_time = time.time()
                trainInfoList.append([epoch, i, loss.item()])
                print('Epoch [{}/{}], Step [{}/{}]'
                      .format(epoch, Epochs, str(i).zfill(3), str(steps_per_epoch).zfill(3)))
            if i % args.ckpt_step == 0 and args.local_rank == 0:
                try:
                    torch.save(model.state_dict(), os.path.join(
                        args.ckpt_path, 'model-{}-{}.ckpt'.format(epoch + 1, i + 1)))
                except:
                    pass
        if args.cycle is False:
            scheduler.step()

        model.eval()
        # mAP_score = validate_multi(val_loader, model, ema, W, device)
        

def validate_multi(val_loader, model, ema_model, W, device):
    print("starting validation")
    Sig = torch.nn.Sigmoid()
    preds_regular = []
    preds_ema = []
    targets = []
    for i, (input, target) in enumerate(val_loader):
        if i % 10 == 0: print(i)
        if i > 100: break
        target = target
        # target = target.max(dim=1)[0]
        
        # compute output
        with torch.no_grad():
            with autocast():
                output_regular = Sig(model(input.to(device))).cpu()
                output_ema = Sig(ema_model.module(input.to(device))).cpu()

        # for mAP calculation
        preds_regular.append(output_regular.cpu().detach())
        preds_ema.append(output_ema.cpu().detach())
        targets.append(target.cpu().detach())
    mAP_score_regular = mAP(torch.cat(targets).numpy(), torch.cat(preds_regular).numpy())
   
    print("mAP score regular {:.2f} ".format(mAP_score_regular))
    return mAP_score_regular


def config_print(rank, batch_size, world_size):
    print('----Torch Config----')
    print('rank : {}'.format(rank))
    print('mini batch-size : {}'.format(batch_size))
    print('world-size : {}'.format(world_size))
    print('backend : {}'.format(args.backend))
    print('--------------------')


def run(rank, batch_size, world_size):
    """ Distributed Synchronous SGD Example """
    config_print(rank, batch_size, world_size)
    main()



def init_processes(rank, world_size, fn, batch_size, backend='gloo'):
    import os

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    # os.environ['NCCL_DEBUG'] = 'INFO'
    # os.environ['GLOO_SOCKET_IFNAME'] = 'eth0'
    dist.init_process_group(backend=backend, init_method="env://")
    fn(rank, batch_size, world_size)
if __name__ == '__main__':
    main()
