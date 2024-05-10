from builtins import breakpoint
import os
import argparse
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from src.helper_functions.helper_functions import mAP, CocoDetection, CutoutPIL, ModelEma, add_weight_decay, OpenImageW, mAP_whitelist, mAP_whitelist_sub, OpenImage, intersection
from src.models import create_model
from src.loss_functions.losses import  AsymmetricLossCustomMS, AsymmetricLossCustomPriorityRankNewNegLand, AsymmetricLossCustomPriorityRankNewPriority, \
    AsymmetricLossCustomPriorityRankNew, AsymmetricLossOrig, AsymmetricLossCustomPriorityRankNewNeg, AsymmetricLossOrigNew
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
parser.add_argument('--ms', action="store_true")
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--limit', type=int, default=15)
parser.add_argument('--gamma_neg', type=int, default=4)
parser.add_argument('--gamma_pos', type=int, default=4)

parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--alpha1', type=float, default=7)
parser.add_argument('--alpha_other', type=float, default=7)
parser.add_argument('--alpha3', type=float, default=0.5)
parser.add_argument('--stop_epoch', type=int, default=5)
parser.add_argument('--alpha5', type=int, default=2)

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
parser.add_argument('--plateau', action='store_true')
parser.add_argument('--weight_balancing', action="store_true")
parser.add_argument('--model_path_openimages', type=str, default='/data/yuhanl/Open_ImagesV6_TRresNet_L_448.pth')
parser.add_argument('--optim', action="store_true")
parser.add_argument('--wl_path', type=str, default='/data/yuhanl/data/yuhanl/Open_ImagesV6_TRresNet_L_448.pth')
parser.add_argument('--ood_weights', type=str, default=None)

parser.add_argument('--neg', action="store_true")
parser.add_argument('--land', action="store_true")


args = parser.parse_args()


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
        else:

        # if len(intersection(all_whitelist_classes, all_neg_classes_human_read)) > 0:
            stats_dict['other'] += 1
    return stats_dict
def main():
    args.do_bottleneck_head = True
    # mapping_dict = { 'clothing': ['Clothing'], 'shirt': ['Shirt'], 'pants': ['Pantsuit'], 'jacket': ['Jacket'], 'footwear': ['Footwear'], 'shoe': ['Shoe'], 'paper': [ 'Paper bag', 'Paper product', 'Paper towel', 'Paper'], 'glass': ['Glass'], 'carton': ['Carton'], 'cardboard': ['Cardboard'], 'tin': ['Tin can', 'Tin'], 'metal': [ 'Metal'], 'plastic': ['Plastic arts', 'Plastic bag', 'Plastic bottle', 'Plastic wrap', 'Plastic']}
    # # mapping_dict = {'paper': ['art paper', 'construction paper', 'household paper product', 'origami paper', 'paper bag', 'paper lantern', 'paper product', 'paper towel', 'paper', 'photographic paper', 'rice paper', 'tissue paper', 'toilet paper', 'wrapping paper'], 'glass': ['beer glass', 'glass bottle', 'glass', 'highball glass', 'magnifying glass', 'martini glass', 'old fashioned glass', 'pint glass', 'shot glass', 'stained glass', 'wine glass'], 'carton': ['carton'], 'cardboard': ['cardboard'], 'tin': ['tin can', 'tin'], 'metal': ['foil (metal)', 'metal', 'metallophone', 'metalsmith', 'metalworking hand tool', 'metalworking'], 'plastic': ['plastic arts', 'plastic bag', 'plastic bottle', 'plastic wrap', 'plastic'], }

    state = torch.load(args.model_path_openimages, map_location='cpu')
    
    whitelist_mapping = {}
    wl_mapping = pd.read_csv(args.wl_path)
    all_classes_count = {}

    for i in range(len(wl_mapping)):
        if wl_mapping.iloc[i]['wl'] not in whitelist_mapping:
            whitelist_mapping[wl_mapping.iloc[i]['wl']] = []
            all_classes_count[wl_mapping.iloc[i]['wl']] = 0
        whitelist_mapping[wl_mapping.iloc[i]['wl']].append(wl_mapping.iloc[i]['class_name'])
    mapping_dict = whitelist_mapping    
    print(mapping_dict)

    class_list = list(state['idx_to_class'].values())
    class_list = [i.replace("'", "").replace("\"", "") for i in class_list]
    
    print('creating model...')

    args.mapping_dict = mapping_dict
    args.do_bottleneck_head = True
    model = create_model(args)

    # for name, param in model.named_parameters():
    if args.model_path:  # make sure to load pretrained ImageNet model
        
        state = torch.load(args.model_path, map_location='cpu')
        # pretrained_dict = {k: v for k, v in state.items() if k in model.state_dict() and 'head' not in k}
        
        model.load_state_dict(state, strict=True)
    


    # # model.head.fc = bottleneck_head(model.num_features, len(W)+1)
    # if args.frozen:
    #     for name, param in model.named_parameters():
    #         if "body" in name:
    #             param.requires_grad = False
    #         if "body.layer4" in name:
    #             param.requires_grad = True
            # if "body.layer3" in name:
            #     param.requires_grad = True
    if args.frozen_all:
        for name, param in model.named_parameters():
            if "body" in name:
                param.requires_grad = False


    print('done\n')
    all_size = len(pd.read_csv(args.data))
    val_size =  int(all_size * 0.1)
    mid_to_human_class_file = pd.read_csv("oidv6-class-descriptions.csv")
    mid_to_human_class = {}
    for i in range(len(mid_to_human_class_file)):
        mid_to_human_class[mid_to_human_class_file.iloc[i][0]] = mid_to_human_class_file.iloc[i][1]
    stats = read_stats(args.data, whitelist_mapping, mid_to_human_class)
    print("Running stats! !!!!!!", stats)
    whole_dataset = OpenImageW(args.root, args.data, args.wl_path,
                                transforms.Compose([
                                    transforms.Resize((args.image_size, args.image_size)),
                                    transforms.ToTensor(),
                                    # normalize, # no need, toTensor does normalization
                                ]), start_idx = 0, end_idx = all_size)
    torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda', args.local_rank)
    model.to(device)
    print("LOCAL RANK: ", args.local_rank)
    print(device)


    print("len(train_dataset)): ", len(whole_dataset))

    # Pytorch Data loader
    train_loader = torch.utils.data.DataLoader(
        whole_dataset, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, shuffle = True)
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=args.batch_size,
    #     num_workers=0, pin_memory=True, sampler=train_sampler)

    # Actuall Training
    train_multi_label_coco(args, model, train_loader, train_loader, args.lr, None, None, device, mapping_dict, args.limit, args.alpha, stats)


def train_multi_label_coco(args, model, train_loader, val_loader, lr, W, W_human, device, mapping_dict,limit, alpha, stats):
    ema = ModelEma(model, 0.9997)  # 0.9997^641=0.82

    # set optimizer
    Epochs = args.epochs
    Stop_epoch = args.stop_epoch
    weight_decay = 1e-4
    print("TRAINING WITH CUSTOM LOSS FUNCTION")
    if args.priority:
        criterion = AsymmetricLossCustomPriorityRankNewPriority(stats=stats, gamma_neg=args.gamma_neg, gamma_pos=args.gamma_pos, alpha5=args.alpha5, alpha=args.alpha, clip=0.05, disable_torch_grad_focal_loss=True, W=W, W_human=W_human, mapping_dict=mapping_dict, limit=limit, asymm = args.asymm, alpha1=args.alpha1, alpha2=args.alpha2, alpha3 = args.alpha3, penalize_other = args.penalize_other, alpha_other=args.alpha_other, weight = args.weight_balancing, sigmoid=args.sigmoid, ood_weights=args.ood_weights)
    elif args.ranklossnew:
        criterion = AsymmetricLossCustomPriorityRankNew(gamma_neg=args.gamma_neg, gamma_pos=args.gamma_pos, alpha5=args.alpha5, clip=0.05, disable_torch_grad_focal_loss=True, W=W, W_human=W_human, mapping_dict=mapping_dict, limit=limit, asymm = args.asymm, alpha1=args.alpha1, alpha2=args.alpha2, alpha3 = args.alpha3, penalize_other = args.penalize_other, alpha_other=args.alpha_other, weight = args.weight_balancing, sigmoid=args.sigmoid)
    elif args.neg:
        criterion = AsymmetricLossCustomPriorityRankNewNeg(stats=stats, gamma_neg=args.gamma_neg, gamma_pos=args.gamma_pos, alpha5=args.alpha5, alpha=args.alpha, clip=0.05, disable_torch_grad_focal_loss=True, W=W, W_human=W_human, mapping_dict=mapping_dict, limit=limit, asymm = args.asymm, alpha1=args.alpha1, alpha2=args.alpha2, alpha3 = args.alpha3, penalize_other = args.penalize_other, alpha_other=args.alpha_other, weight = args.weight_balancing, sigmoid=args.sigmoid, ood_weights=args.ood_weights)
    elif args.land:
        criterion = AsymmetricLossCustomPriorityRankNewNegLand(stats=stats, gamma_neg=args.gamma_neg, gamma_pos=args.gamma_pos, alpha5=args.alpha5, alpha=args.alpha, clip=0.05, disable_torch_grad_focal_loss=True, W=W, W_human=W_human, mapping_dict=mapping_dict, limit=limit, asymm = args.asymm, alpha1=args.alpha1, alpha2=args.alpha2, alpha3 = args.alpha3, penalize_other = args.penalize_other, alpha_other=args.alpha_other, weight = args.weight_balancing, sigmoid=args.sigmoid, ood_weights=args.ood_weights)
    elif args.ms:
        criterion = AsymmetricLossCustomMS(stats=stats, gamma_neg=args.gamma_neg, gamma_pos=args.gamma_pos, alpha5=args.alpha5, alpha=args.alpha, clip=0.05, disable_torch_grad_focal_loss=True, W=W, W_human=W_human, mapping_dict=mapping_dict, limit=limit, asymm = args.asymm, alpha1=args.alpha1, alpha2=args.alpha2, alpha3 = args.alpha3, penalize_other = args.penalize_other, alpha_other=args.alpha_other, weight = args.weight_balancing, sigmoid=args.sigmoid, ood_weights=args.ood_weights)
    else:
        criterion = AsymmetricLossOrigNew(mapping_dict=mapping_dict, stats=stats, gamma_neg=args.gamma_neg, gamma_pos=args.gamma_pos, clip=0.05, disable_torch_grad_focal_loss=True, ood_weights=args.ood_weights)

    parameters = add_weight_decay(model, weight_decay)
    optimizer = torch.optim.Adam(params=parameters, lr=lr, weight_decay=0)  # true wd, filter_bias_and_bn
    steps_per_epoch = len(train_loader)
    if args.cycle:
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=Epochs,
                                            pct_start=0.2)
    elif args.plateau:
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    else:
        scheduler = lr_scheduler.StepLR(optimizer, step_size = 1, gamma=0.1)
    highest_mAP = 0
    trainInfoList = []
    scaler = GradScaler()
    start_time = time.time()
    if os.path.exists(args.ckpt_path) is False:
        os.makedirs(args.ckpt_path)
    for epoch in range(Epochs):
        if epoch > Stop_epoch:
            break

        for i, (inputData, target, target_neg) in enumerate(train_loader):
            model.train()
            inputData = inputData.to(device)
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
            if args.plateau:
                scheduler.step(loss)
            # optimizer.step()


            ema.update(model)
            # store information
            if i % 100 == 0 and args.rank == 0:
                start_time = time.time()
                trainInfoList.append([epoch, i, loss.item()])
                print('Epoch [{}/{}], Step [{}/{}]'
                      .format(epoch, Epochs, str(i).zfill(3), str(steps_per_epoch).zfill(3), \
                              loss.item()))
            if i % args.ckpt_step == 0 and args.local_rank == 0:
                try:
                    torch.save(model.state_dict(), os.path.join(
                        args.ckpt_path, 'model-{}-{}.ckpt'.format(epoch + 1, i + 1)))
                except:
                    pass
        if args.cycle is False and args.plateau is False:
            scheduler.step()

        # model.eval()
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
