import torch
import torch.nn as nn
import numpy as np
import copy

import torch.nn.functional as F
import pandas as pd
def calculate_active_ind(active_ind, ind):
    if active_ind is None:
        active_ind = ind
    else:
        active_ind = np.concatenate((active_ind, ind))
    return active_ind


class AsymmetricLossOrigNew(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLossOrigNew, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y, y_neg):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y )  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y-y_neg) + self.gamma_pos * y_neg
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()

class AsymmetricLossOrig(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLossOrig, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()

class AsymmetricLossCustom(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True,W=None, W_human=None, mapping_dict=None, limit=10, asymm=False, alpha=None):
        super(AsymmetricLossCustom, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.W = W
        state = torch.load("/data/yuhanl/Open_ImagesV6_TRresNet_L_448.pth", map_location='cpu')
        self.class_list = list(state['idx_to_class'].values())
        self.class_list = [i.replace("'", "").replace("\"", "") for i in self.class_list]
        recycle_labels = ['plastic', 'glass', 'paper', 'cardboard', 'metal', 'tin', 'carton']
        donate_labels = ['clothes', 'clothing', 'shirt', 'pants', 'jacket', 'footwear', 'shoe']
        compost_labels = ['food','snack','compost']
        self.asymm = asymm
        self.recycle_w_gt_ind = []
        for label in recycle_labels:
            if label not in mapping_dict:
                continue
            for value in mapping_dict[label][:limit]:
                self.recycle_w_gt_ind.append(W_human.index(value))
        self.donate_w_gt_ind = []
        for label in donate_labels:
            if label not in mapping_dict:
                continue
            for value in mapping_dict[label][:limit]:
                self.donate_w_gt_ind.append(W_human.index(value))
        self.donate_w_gt_ind = np.array(self.donate_w_gt_ind)
        self.recycle_w_gt_ind = np.array(self.recycle_w_gt_ind)

        self.compost_w_gt_ind = []
        for label in compost_labels:
            if label not in mapping_dict:
                continue
            for value in mapping_dict[label][:limit]:
                self.compost_w_gt_ind.append(W_human.index(value))
        self.compost_w_gt_ind = np.array(self.compost_w_gt_ind)

        self.mapping_dict = mapping_dict
        self.alpha = alpha
        print("ALPHA IS: ", alpha)

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        y_np = y.cpu().numpy()

        loss = torch.zeros(y.shape).cuda()
        for i in range(y.shape[0]):
            y_telda = np.zeros(y[i].shape)
            help_y_recycle = y_np[i][self.recycle_w_gt_ind]
            help_y_donate = y_np[i][self.donate_w_gt_ind]
            help_y_compost = y_np[i][self.compost_w_gt_ind]
            y_telda = y_np[i]
            active_ind = None # all the indices except the indices in the whitelist
            whitelist_ind = None
            active_whitelist = [0,0,0]
            if np.sum( help_y_recycle) > 0:
                y_telda[self.recycle_w_gt_ind] = 1
                if whitelist_ind is None:
                    whitelist_ind = self.recycle_w_gt_ind
            if np.sum(help_y_donate ) > 0:
                y_telda[self.donate_w_gt_ind] = 1

                if whitelist_ind is None:
                    whitelist_ind = self.donate_w_gt_ind
                else:
                    whitelist_ind = np.concatenate((whitelist_ind, self.donate_w_gt_ind))
                # print("dontae")
            if np.sum(help_y_compost) > 0:
                # print("Compost")
                y_telda[self.compost_w_gt_ind] = 1
                if whitelist_ind is None:
                    whitelist_ind = self.compost_w_gt_ind
                else:
                    whitelist_ind = np.concatenate((whitelist_ind, self.compost_w_gt_ind))
            if whitelist_ind is not None:
                if np.sum( help_y_recycle) == 0:
                    active_ind = calculate_active_ind(active_ind, self.recycle_w_gt_ind)
                if np.sum( help_y_donate) == 0:
                    active_ind = calculate_active_ind(active_ind, self.donate_w_gt_ind)
                if np.sum(help_y_compost) == 0:
                    active_ind = calculate_active_ind(active_ind, self.compost_w_gt_ind)

            loss_orig = y[i] * torch.log(xs_pos[i].clamp(min=self.eps)) + (1 - y[i]) * torch.log(xs_neg[i].clamp(min=self.eps))

            loss[i] = loss_orig
            if active_ind is not None:
                loss[i][active_ind] = loss[i][active_ind] * self.alpha

            # if whitelist_ind is not None:
            #     loss[i][whitelist_ind] = torch.max(loss_orig[whitelist_ind], loss_telda[whitelist_ind])
        # los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        # los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        # loss_bce = los_pos + los_neg
        # print("Our loss: {}, normal loss: {}".format(str(-loss.sum().item()), str(-loss_bce.sum().item())))
        # los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        # los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        # loss = los_pos + los_neg

        # Asymmetric Focusing
        # if self.asymm and ( self.gamma_neg > 0 or self.gamma_pos > 0) :
        #     if self.disable_torch_grad_focal_loss:
        #         torch.set_grad_enabled(False)
        #     pt0 = xs_pos * y
        #     pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
        #     pt = pt0 + pt1
        #     one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
        #     one_sided_w = torch.pow(1 - pt, one_sided_gamma)
        #     if self.disable_torch_grad_focal_loss:
        #         torch.set_grad_enabled(True)
        #     loss *= one_sided_w

        return -loss.sum()

def find_whitelist(index, compost_w_gt_ind, recycle_w_gt_ind, donate_w_gt_ind):
    if index in compost_w_gt_ind:
        return 1
    if index in recycle_w_gt_ind:
        return 2
    if index in donate_w_gt_ind:
        return 3
    return 4


class AsymmetricLossCustomPriorityFinalAll(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True,W=None, W_human=None, mapping_dict=None, limit=10, asymm=False, alpha1=None, alpha2=None, penalize_other=False, alpha_other=None, weight=False):
        super(AsymmetricLossCustomPriorityFinalAll, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.W = W
        self.W_human = W_human
        state = torch.load("/data/yuhanl/Open_ImagesV6_TRresNet_L_448.pth", map_location='cpu')
        self.class_list = list(state['idx_to_class'].values())
        self.class_list = [i.replace("'", "").replace("\"", "") for i in self.class_list]
        recycle_labels = ['plastic', 'glass', 'paper', 'cardboard', 'metal', 'tin', 'carton']
        donate_labels = ['clothes', 'clothing', 'shirt', 'pants', 'jacket', 'footwear', 'shoe']
        compost_labels = ['food','snack','compost']
        self.asymm = asymm
        self.recycle_w_gt_ind = []
        for label in recycle_labels:
            if label not in mapping_dict:
                continue
            for value in mapping_dict[label][:limit]:
                self.recycle_w_gt_ind.append(self.class_list.index(value))

        self.donate_w_gt_ind = []
        for label in donate_labels:
            if label not in mapping_dict:
                continue
            for value in mapping_dict[label][:limit]:
                self.donate_w_gt_ind.append(self.class_list.index(value))
        self.donate_w_gt_ind = np.array(self.donate_w_gt_ind)
        self.recycle_w_gt_ind = np.array(self.recycle_w_gt_ind)

        self.compost_w_gt_ind = []
        for label in compost_labels:
            if label not in mapping_dict:
                continue
            for value in mapping_dict[label][:limit]:
                self.compost_w_gt_ind.append(self.class_list.index(value))
        self.compost_w_gt_ind = np.array(self.compost_w_gt_ind)

        self.mapping_dict = mapping_dict
        self.alpha1 = alpha1
        self.alpha2 = alpha2

        self.penalize_other = penalize_other
        self.alpha_other = alpha_other
        print("PRIORITY LOSS!!!")
        print("penalize_other: ", self.penalize_other)
        print("recycle index: ", self.recycle_w_gt_ind)
        print("donate index: ", self.donate_w_gt_ind)
        print("compost index: ", self.compost_w_gt_ind)
        print(self.alpha1)

        w_human_lower = [i.lower().split(" ")[0] for i in W_human]

        self.whitelist_indices = [self.compost_w_gt_ind, self.recycle_w_gt_ind, self.donate_w_gt_ind, np.array([len(W_human)])]
        self.whitelist_mapping = {}

        for i in range(len(W_human)):
            self.whitelist_mapping[self.class_list.index(W_human[i])] = find_whitelist(self.class_list.index(W_human[i]), self.compost_w_gt_ind, self.recycle_w_gt_ind, self.donate_w_gt_ind)
        print(self.whitelist_mapping)
        self.priority_list = [1, 2, 3, 4]
    def find_w_ranking(self, arg_sort):
        w_dict = {}
        arg_sort = list(arg_sort.cpu().numpy())
        for w_human in self.W_human:
            index = self.class_list.index(w_human)
            w_dict[w_human] = arg_sort.index(index)
        print(w_dict)
    def find_gt_whitelist(self, j):
        gt_whitelist = []
        min_index = 99999
        if torch.sum(j[self.compost_w_gt_ind] ) > 0:
            gt_whitelist.append(1)
            min_index = min(min_index, self.priority_list.index(1))
        if torch.sum(j[self.recycle_w_gt_ind] ) > 0:
            gt_whitelist.append(2)
            min_index = min(min_index, self.priority_list.index(2))
        if torch.sum(j[self.donate_w_gt_ind] ) > 0:
            gt_whitelist.append(3)
            min_index = min(min_index, self.priority_list.index(3))
        if len(gt_whitelist) == 0:
            gt_whitelist.append(4)
        return gt_whitelist, min_index
    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        y_np = y
        loss = torch.zeros(y.shape).cuda()
        x_sigmoid_np = x_sigmoid
        for i in range(y.shape[0]):
            y_telda = np.zeros(y[i].shape)
            help_y_recycle = y_np[i][self.recycle_w_gt_ind]
            help_y_donate = y_np[i][self.donate_w_gt_ind]
            help_y_compost = y_np[i][self.compost_w_gt_ind]
            y_telda = y_np[i]
            active_ind = None # all the indices except the indices in the whitelist
            whitelist_ind = None
            arg_sort = torch.argsort(x_sigmoid[i], descending=True)
            loss[i] = y[i] * torch.log(xs_pos[i].clamp(min=self.eps)) + (1 - y[i]) * torch.log(xs_neg[i].clamp(min=self.eps))
            first_gt_whitelist_index = -1
            penalty_vector = np.zeros(len(y_np[i]))
            for j in arg_sort[:10]:
                if j not in self.whitelist_mapping:
                    continue
                if j in self.whitelist_mapping and gt_whitelist_index[0] == 4:
                    loss[i][j] *= self.alpha_other
                whitelist_index = self.whitelist_mapping[j.item()]

                if whitelist_index in gt_whitelist_index :
                    first_gt_whitelist_index = j
                    # loss[i][j] *= 0.1
                else:
                    # Penalize greatly
                    if first_gt_whitelist_index == -1:
                        loss[i][j] *= self.alpha1
                        penalty_vector[j] = self.alpha1

            if first_gt_whitelist_index == -1:
                for j in arg_sort[:10]:
                    loss[i][j] *= self.alpha1
                    penalty_vector[j] = self.alpha1

                # print(xs_pos[i])
            # print(gt_whitelist_index)
            # print((y_np[i] == 1).nonzero(as_tuple=True)[0])

            # # print(penalty_vector)
            # # print(gt_whitelist_index)
            # # self.find_w_ranking(arg_sort)

            # if np.sum(penalty_vector == self.alpha1) > 0:
            #     print("Penalized greatly: ", np.sum(penalty_vector == self.alpha1))
            # print("++++++++++++")
        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0 :
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()




class AsymmetricLossCustomPrioritySmallTest(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True,W=None, W_human=None, mapping_dict=None, limit=10, asymm=False, alpha1=None, alpha2=None, alpha3=None, penalize_other=False, alpha_other=None, weight=False):
        super(AsymmetricLossCustomPrioritySmallTest, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.W = W
        self.W_human = W_human
        state = torch.load("/data/yuhanl/Open_ImagesV6_TRresNet_L_448.pth", map_location='cpu')
        self.class_list = list(state['idx_to_class'].values())
        self.class_list = [i.replace("'", "").replace("\"", "") for i in self.class_list]
        recycle_labels = ['plastic', 'glass', 'paper', 'cardboard', 'metal', 'tin', 'carton']
        donate_labels = ['clothes', 'clothing', 'shirt', 'pants', 'jacket', 'footwear', 'shoe']
        compost_labels = ['food','snack','compost']
        self.asymm = asymm
        self.recycle_w_gt_ind = []
        for label in recycle_labels:
            if label not in mapping_dict:
                continue
            for value in mapping_dict[label][:limit]:
                self.recycle_w_gt_ind.append(self.class_list.index(value))

        self.donate_w_gt_ind = []
        for label in donate_labels:
            if label not in mapping_dict:
                continue
            for value in mapping_dict[label][:limit]:
                self.donate_w_gt_ind.append(self.class_list.index(value))
        self.donate_w_gt_ind = np.array(self.donate_w_gt_ind)
        self.recycle_w_gt_ind = np.array(self.recycle_w_gt_ind)

        self.compost_w_gt_ind = []
        for label in compost_labels:
            if label not in mapping_dict:
                continue
            for value in mapping_dict[label][:limit]:
                self.compost_w_gt_ind.append(self.class_list.index(value))
        self.compost_w_gt_ind = np.array(self.compost_w_gt_ind)

        self.mapping_dict = mapping_dict
        self.alpha1 = alpha1
        self.alpha2 = alpha2

        self.penalize_other = penalize_other
        self.alpha_other = alpha_other
        print("PRIORITY LOSS!!!")
        print("penalize_other: ", self.penalize_other)
        print("recycle index: ", self.recycle_w_gt_ind)
        print("donate index: ", self.donate_w_gt_ind)
        print("compost index: ", self.compost_w_gt_ind)
        print(self.alpha1)

        w_human_lower = [i.lower().split(" ")[0] for i in W_human]

        self.whitelist_indices = [self.compost_w_gt_ind, self.recycle_w_gt_ind, self.donate_w_gt_ind, np.array([len(W_human)])]
        self.whitelist_mapping = {}

        for i in range(len(W_human)):
            self.whitelist_mapping[self.class_list.index(W_human[i])] = find_whitelist(self.class_list.index(W_human[i]), self.compost_w_gt_ind, self.recycle_w_gt_ind, self.donate_w_gt_ind)
        print(self.whitelist_mapping)
        self.priority_list = [1, 2, 3, 4]
        self.alpha3 = alpha3
    def find_w_ranking(self, arg_sort):
        w_dict = {}
        arg_sort = list(arg_sort.cpu().numpy())
        for w_human in self.W_human:
            index = self.class_list.index(w_human)
            w_dict[w_human] = arg_sort.index(index)
        print(w_dict)
    def find_gt_whitelist(self, j):
        gt_whitelist = []
        min_index = 99999
        if torch.sum(j[self.compost_w_gt_ind] ) > 0:
            gt_whitelist.append(1)
            min_index = min(min_index, self.priority_list.index(1))
        if torch.sum(j[self.recycle_w_gt_ind] ) > 0:
            gt_whitelist.append(2)
            min_index = min(min_index, self.priority_list.index(2))
        if torch.sum(j[self.donate_w_gt_ind] ) > 0:
            gt_whitelist.append(3)
            min_index = min(min_index, self.priority_list.index(3))
        if len(gt_whitelist) == 0:
            gt_whitelist.append(4)
        return gt_whitelist, min_index
    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        y_np = y
        loss = torch.zeros(y.shape).cuda()
        x_sigmoid_np = x_sigmoid
        for i in range(y.shape[0]):
            y_telda = np.zeros(y[i].shape)
            help_y_recycle = y_np[i][self.recycle_w_gt_ind]
            help_y_donate = y_np[i][self.donate_w_gt_ind]
            help_y_compost = y_np[i][self.compost_w_gt_ind]
            y_telda = y_np[i]
            active_ind = None # all the indices except the indices in the whitelist
            whitelist_ind = None
            arg_sort = torch.argsort(x_sigmoid[i], descending=True)
            loss[i] = y[i] * torch.log(xs_pos[i].clamp(min=self.eps)) + (1 - y[i]) * torch.log(xs_neg[i].clamp(min=self.eps))
            first_gt_whitelist_index = -1
            gt_whitelist_index, min_index = self.find_gt_whitelist(y_np[i])

            penalty_vector = np.zeros(len(y_np[i]))

            for j in arg_sort[:10]:
                index = j.item()
                if index not in self.whitelist_mapping:
                    # Penalize less if errors in correct Other
                    if gt_whitelist_index[0] == 4:
                        loss[i][index] *= self.alpha3
                        penalty_vector[index] = self.alpha3
                else:
                    if y[i][index] == 0:
                        loss[i][index] *= self.alpha3
                        penalty_vector[index] = self.alpha3


        if self.gamma_neg > 0 or self.gamma_pos > 0 :
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()


class AsymmetricLossCustomPrioritySmallFocal(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True,W=None, W_human=None, mapping_dict=None, limit=10, asymm=False, alpha1=None, alpha2=None, alpha3=None, penalize_other=False, alpha_other=None, weight=False):
        super(AsymmetricLossCustomPrioritySmallFocal, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.W = W
        self.W_human = W_human
        state = torch.load("/data/yuhanl/Open_ImagesV6_TRresNet_L_448.pth", map_location='cpu')
        self.class_list = list(state['idx_to_class'].values())
        self.class_list = [i.replace("'", "").replace("\"", "") for i in self.class_list]
        recycle_labels = ['plastic', 'glass', 'paper', 'cardboard', 'metal', 'tin', 'carton']
        donate_labels = ['clothes', 'clothing', 'shirt', 'pants', 'jacket', 'footwear', 'shoe']
        compost_labels = ['food','snack','compost']
        self.asymm = asymm
        self.recycle_w_gt_ind = []
        for label in recycle_labels:
            if label not in mapping_dict:
                continue
            for value in mapping_dict[label][:limit]:
                self.recycle_w_gt_ind.append(self.class_list.index(value))

        self.donate_w_gt_ind = []
        for label in donate_labels:
            if label not in mapping_dict:
                continue
            for value in mapping_dict[label][:limit]:
                self.donate_w_gt_ind.append(self.class_list.index(value))
        self.donate_w_gt_ind = np.array(self.donate_w_gt_ind)
        self.recycle_w_gt_ind = np.array(self.recycle_w_gt_ind)

        self.compost_w_gt_ind = []
        for label in compost_labels:
            if label not in mapping_dict:
                continue
            for value in mapping_dict[label][:limit]:
                self.compost_w_gt_ind.append(self.class_list.index(value))
        self.compost_w_gt_ind = np.array(self.compost_w_gt_ind)

        self.mapping_dict = mapping_dict
        self.alpha1 = alpha1
        self.alpha2 = alpha2

        self.penalize_other = penalize_other
        self.alpha_other = alpha_other
        print("PRIORITY LOSS!!!")
        print("penalize_other: ", self.penalize_other)
        print("recycle index: ", self.recycle_w_gt_ind)
        print("donate index: ", self.donate_w_gt_ind)
        print("compost index: ", self.compost_w_gt_ind)
        print(self.alpha1)

        w_human_lower = [i.lower().split(" ")[0] for i in W_human]

        self.whitelist_indices = [self.compost_w_gt_ind, self.recycle_w_gt_ind, self.donate_w_gt_ind, np.array([len(W_human)])]
        self.whitelist_mapping = {}

        for i in range(len(W_human)):
            self.whitelist_mapping[self.class_list.index(W_human[i])] = find_whitelist(self.class_list.index(W_human[i]), self.compost_w_gt_ind, self.recycle_w_gt_ind, self.donate_w_gt_ind)
        print(self.whitelist_mapping)
        self.priority_list = [1, 2, 3, 4]
        self.alpha3 = alpha3
        print("Alpha3 : alpha3")
    def find_w(self, j):
        if j in self.compost_w_gt_ind:
            return 1
        if j in self.recycle_w_gt_ind:
            return 2
        if j in self.donate_w_gt_ind:
            return 3
        return 4
    def find_gt_whitelist(self, j):
        gt_whitelist = []
        min_index = 99999
        if torch.sum(j[self.compost_w_gt_ind] ) > 0:
            gt_whitelist.append(1)
            min_index = min(min_index, self.priority_list.index(1))
        if torch.sum(j[self.recycle_w_gt_ind] ) > 0:
            gt_whitelist.append(2)
            min_index = min(min_index, self.priority_list.index(2))
        if torch.sum(j[self.donate_w_gt_ind] ) > 0:
            gt_whitelist.append(3)
            min_index = min(min_index, self.priority_list.index(3))
        if len(gt_whitelist) == 0:
            gt_whitelist.append(4)
        return gt_whitelist, min_index
    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        y_np = y
        loss = torch.zeros(y.shape).cuda()
        x_sigmoid_np = x_sigmoid
        for i in range(y.shape[0]):
            y_telda = np.zeros(y[i].shape)
            help_y_recycle = y_np[i][self.recycle_w_gt_ind]
            help_y_donate = y_np[i][self.donate_w_gt_ind]
            help_y_compost = y_np[i][self.compost_w_gt_ind]
            y_telda = y_np[i]
            active_ind = None # all the indices except the indices in the whitelist
            whitelist_ind = None
            arg_sort = torch.argsort(x_sigmoid[i], descending=True)
            loss[i] = y[i] * torch.log(xs_pos[i].clamp(min=self.eps)) + (1 - y[i]) * torch.log(xs_neg[i].clamp(min=self.eps))
            first_gt_whitelist_index = -1
            gt_whitelist_index, min_index = self.find_gt_whitelist(y_np[i])

            penalty_vector = np.zeros(len(y_np[i]))

            for j in arg_sort[:10]:
                index = j.item()
                wl_index = self.find_w(index)

                if index not in self.whitelist_mapping and gt_whitelist_index[0] == 4:
                    # Penalize less if errors in correct Other
                    if y_np[i][index] == 0:
                        loss[i][index] *= xs_pos[i][index] * self.alpha3
                    else:
                        loss[i][index] *= xs_neg[i][index] * self.alpha3
                elif wl_index in gt_whitelist_index:
                    if y_np[i][index] == 0:
                        loss[i][index] *= xs_pos[i][index] * self.alpha3
                    else:
                        loss[i][index] *= xs_neg[i][index] * self.alpha3

            # if np.sum(penalty_vector == self.alpha3) > 0:
            #     print("Penalized greatly: ", np.sum(penalty_vector == self.alpha3))
            #     print("++++++++++++")

        if self.gamma_neg > 0 or self.gamma_pos > 0 :
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()


class AsymmetricLossCustomPrioritySmall(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True,W=None, W_human=None, mapping_dict=None, limit=10, asymm=False, alpha1=None, alpha2=None, alpha3=None, penalize_other=False, alpha_other=None, weight=False):
        super(AsymmetricLossCustomPrioritySmall, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.W = W
        self.W_human = W_human
        state = torch.load("/data/yuhanl/Open_ImagesV6_TRresNet_L_448.pth", map_location='cpu')
        self.class_list = list(state['idx_to_class'].values())
        self.class_list = [i.replace("'", "").replace("\"", "") for i in self.class_list]
        recycle_labels = ['plastic', 'glass', 'paper', 'cardboard', 'metal', 'tin', 'carton']
        donate_labels = ['clothes', 'clothing', 'shirt', 'pants', 'jacket', 'footwear', 'shoe']
        compost_labels = ['food','snack','compost']
        self.asymm = asymm
        self.recycle_w_gt_ind = []
        for label in recycle_labels:
            if label not in mapping_dict:
                continue
            for value in mapping_dict[label][:limit]:
                self.recycle_w_gt_ind.append(self.class_list.index(value))

        self.donate_w_gt_ind = []
        for label in donate_labels:
            if label not in mapping_dict:
                continue
            for value in mapping_dict[label][:limit]:
                self.donate_w_gt_ind.append(self.class_list.index(value))
        self.donate_w_gt_ind = np.array(self.donate_w_gt_ind)
        self.recycle_w_gt_ind = np.array(self.recycle_w_gt_ind)

        self.compost_w_gt_ind = []
        for label in compost_labels:
            if label not in mapping_dict:
                continue
            for value in mapping_dict[label][:limit]:
                self.compost_w_gt_ind.append(self.class_list.index(value))
        self.compost_w_gt_ind = np.array(self.compost_w_gt_ind)

        self.mapping_dict = mapping_dict
        self.alpha1 = alpha1
        self.alpha2 = alpha2

        self.penalize_other = penalize_other
        self.alpha_other = alpha_other
        print("PRIORITY LOSS!!!")
        print("penalize_other: ", self.penalize_other)
        print("recycle index: ", self.recycle_w_gt_ind)
        print("donate index: ", self.donate_w_gt_ind)
        print("compost index: ", self.compost_w_gt_ind)
        print(self.alpha1)

        w_human_lower = [i.lower().split(" ")[0] for i in W_human]

        self.whitelist_indices = [self.compost_w_gt_ind, self.recycle_w_gt_ind, self.donate_w_gt_ind, np.array([len(W_human)])]
        self.whitelist_mapping = {}

        for i in range(len(W_human)):
            self.whitelist_mapping[self.class_list.index(W_human[i])] = find_whitelist(self.class_list.index(W_human[i]), self.compost_w_gt_ind, self.recycle_w_gt_ind, self.donate_w_gt_ind)
        print(self.whitelist_mapping)
        self.priority_list = [1, 2, 3, 4]
        self.alpha3 = alpha3
        print("Alpha3 : alpha3")
    def find_w(self, j):
        if j in self.compost_w_gt_ind:
            return 1
        if j in self.recycle_w_gt_ind:
            return 2
        if j in self.donate_w_gt_ind:
            return 3
        return 4
    def find_gt_whitelist(self, j):
        gt_whitelist = []
        min_index = 99999
        if torch.sum(j[self.compost_w_gt_ind] ) > 0:
            gt_whitelist.append(1)
            min_index = min(min_index, self.priority_list.index(1))
        if torch.sum(j[self.recycle_w_gt_ind] ) > 0:
            gt_whitelist.append(2)
            min_index = min(min_index, self.priority_list.index(2))
        if torch.sum(j[self.donate_w_gt_ind] ) > 0:
            gt_whitelist.append(3)
            min_index = min(min_index, self.priority_list.index(3))
        if len(gt_whitelist) == 0:
            gt_whitelist.append(4)
        return gt_whitelist, min_index
    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        y_np = y
        loss = torch.zeros(y.shape).cuda()
        x_sigmoid_np = x_sigmoid
        for i in range(y.shape[0]):
            y_telda = np.zeros(y[i].shape)
            help_y_recycle = y_np[i][self.recycle_w_gt_ind]
            help_y_donate = y_np[i][self.donate_w_gt_ind]
            help_y_compost = y_np[i][self.compost_w_gt_ind]
            y_telda = y_np[i]
            active_ind = None # all the indices except the indices in the whitelist
            whitelist_ind = None
            arg_sort = torch.argsort(x_sigmoid[i], descending=True)
            loss[i] = y[i] * torch.log(xs_pos[i].clamp(min=self.eps)) + (1 - y[i]) * torch.log(xs_neg[i].clamp(min=self.eps))
            first_gt_whitelist_index = -1
            gt_whitelist_index, min_index = self.find_gt_whitelist(y_np[i])

            penalty_vector = np.zeros(len(y_np[i]))

            for j in arg_sort[:10]:
                index = j.item()
                wl_index = self.find_w(index)
                if index not in self.whitelist_mapping:
                    # Penalize less if errors in correct Other
                    if gt_whitelist_index[0] == 4:
                        if y[i][index] == 0:
                            if self.penalize_other:
                                loss[i][index] *= self.alpha3
                                penalty_vector[index] = self.alpha3
                elif wl_index in gt_whitelist_index:
                    if y[i][index] == 0:
                        loss[i][index] *= self.alpha3
                        penalty_vector[index] = self.alpha3

            # if np.sum(penalty_vector == self.alpha3) > 0:
            #     print("Penalized greatly: ", np.sum(penalty_vector == self.alpha3))
            #     print("++++++++++++")

        if self.gamma_neg > 0 or self.gamma_pos > 0 :
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()


class AsymmetricLossCustomPriorityLargeOld(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True,W=None, W_human=None, mapping_dict=None, limit=10, asymm=False, alpha1=None, alpha2=None, alpha3=None, penalize_other=False, alpha_other=None, weight=False):
        super(AsymmetricLossCustomPriorityLargeOld, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.W = W
        self.W_human = W_human
        state = torch.load("/data/yuhanl/Open_ImagesV6_TRresNet_L_448.pth", map_location='cpu')
        self.class_list = list(state['idx_to_class'].values())
        self.class_list = [i.replace("'", "").replace("\"", "") for i in self.class_list]
        recycle_labels = ['plastic', 'glass', 'paper', 'cardboard', 'metal', 'tin', 'carton']
        donate_labels = ['clothes', 'clothing', 'shirt', 'pants', 'jacket', 'footwear', 'shoe']
        compost_labels = ['food','snack','compost']
        self.asymm = asymm
        self.recycle_w_gt_ind = []
        for label in recycle_labels:
            if label not in mapping_dict:
                continue
            for value in mapping_dict[label][:limit]:
                self.recycle_w_gt_ind.append(self.class_list.index(value))

        self.donate_w_gt_ind = []
        for label in donate_labels:
            if label not in mapping_dict:
                continue
            for value in mapping_dict[label][:limit]:
                self.donate_w_gt_ind.append(self.class_list.index(value))
        self.donate_w_gt_ind = np.array(self.donate_w_gt_ind)
        self.recycle_w_gt_ind = np.array(self.recycle_w_gt_ind)

        self.compost_w_gt_ind = []
        for label in compost_labels:
            if label not in mapping_dict:
                continue
            for value in mapping_dict[label][:limit]:
                self.compost_w_gt_ind.append(self.class_list.index(value))
        self.compost_w_gt_ind = np.array(self.compost_w_gt_ind)

        self.mapping_dict = mapping_dict
        self.alpha1 = alpha1
        self.alpha2 = alpha2

        self.penalize_other = penalize_other
        self.alpha_other = alpha_other
        print("PRIORITY LOSS!!!")
        print("penalize_other: ", self.penalize_other)
        print("recycle index: ", self.recycle_w_gt_ind)
        print("donate index: ", self.donate_w_gt_ind)
        print("compost index: ", self.compost_w_gt_ind)
        print(self.alpha1)

        w_human_lower = [i.lower().split(" ")[0] for i in W_human]

        self.whitelist_indices = [self.compost_w_gt_ind, self.recycle_w_gt_ind, self.donate_w_gt_ind]
        self.whitelist_mapping = {}
        self.total_indices = np.array([True for i in range(len(self.class_list))])
        for item in self.whitelist_indices:
            self.total_indices[item] = False
        for i in range(len(W_human)):
            self.whitelist_mapping[self.class_list.index(W_human[i])] = find_whitelist(self.class_list.index(W_human[i]), self.compost_w_gt_ind, self.recycle_w_gt_ind, self.donate_w_gt_ind)
        print(self.whitelist_mapping)
        self.whitelist_indices_bool = []
        for item in self.whitelist_indices:
            indices =  np.array([False for i in range(len(self.class_list))])
            indices[item] = True
            self.whitelist_indices_bool.append(indices)
        self.priority_list = [1, 2, 3, 4]
        self.alpha3 = alpha3
        print("Large loss!!!")
    def find_w(self, j):
        if j in self.compost_w_gt_ind:
            return 1
        if j in self.recycle_w_gt_ind:
            return 2
        if j in self.donate_w_gt_ind:
            return 3
        return 4
    def find_gt_whitelist(self, j):
        gt_whitelist = []
        min_index = 99999
        if torch.sum(j[self.compost_w_gt_ind] ) > 0:
            gt_whitelist.append(1)
            min_index = min(min_index, self.priority_list.index(1))
        if torch.sum(j[self.recycle_w_gt_ind] ) > 0:
            gt_whitelist.append(2)
            min_index = min(min_index, self.priority_list.index(2))
        if torch.sum(j[self.donate_w_gt_ind] ) > 0:
            gt_whitelist.append(3)
            min_index = min(min_index, self.priority_list.index(3))
        if len(gt_whitelist) == 0:
            gt_whitelist.append(4)
        return gt_whitelist, min_index
    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        y_np = y
        loss = torch.zeros(y.shape).cuda()
        x_sigmoid_np = x_sigmoid
        for i in range(y.shape[0]):
            y_telda = np.zeros(y[i].shape)
            help_y_recycle = y_np[i][self.recycle_w_gt_ind]
            help_y_donate = y_np[i][self.donate_w_gt_ind]
            help_y_compost = y_np[i][self.compost_w_gt_ind]
            y_telda = y_np[i]
            active_ind = None # all the indices except the indices in the whitelist
            whitelist_ind = None
            arg_sort = torch.argsort(x_sigmoid[i], descending=True)
            loss[i] = y[i] * torch.log(xs_pos[i].clamp(min=self.eps)) + (1 - y[i]) * torch.log(xs_neg[i].clamp(min=self.eps))
            first_gt_whitelist_index = -1
            gt_whitelist_index, min_index = self.find_gt_whitelist(y_np[i])
            penalty_vector = torch.ones(len(y_np[i])).cuda() * self.alpha1
            if gt_whitelist_index[0] == 4:
                penalty_vector[self.total_indices] = self.alpha3
                loss[i] *= penalty_vector
            else:
                # for j in range(1, 4):
                #     if j not in gt_whitelist_index:
                #         penalty_vector[self.whitelist_indices_bool[j - 1]] = 1/self.alpha3
                # loss[i] *= penalty_vector
                for gt_index in gt_whitelist_index:
                    max_pred_index = torch.argmax(xs_pos[i][self.whitelist_indices[gt_index-1]])
                    global_max_pred_index = self.whitelist_indices[gt_index-1][max_pred_index]
                    # print(gt_index)
                    # print(self.whitelist_indices[gt_index - 1])
                    for ind in self.whitelist_indices[gt_index - 1]:
                        loss[i][ind] *= self.alpha3
                    loss[i][global_max_pred_index] = torch.log(xs_pos[i][global_max_pred_index].clamp(min=self.eps)) * self.alpha1 * self.alpha2
                    # loss[i][global_max_pred_index] = torch.log(xs_pos[i][global_max_pred_index].clamp(min=self.eps)) * self.alpha1 * 10
                    # print(gt_index)
                    # print(xs_pos[i][self.whitelist_indices[gt_index-1][max_pred_index]])
                    # print(self.class_list[self.whitelist_indices[gt_index-1][max_pred_index]])
                    # print("++++++++++++")
                if self.penalize_other:
                    penalty_vector[self.total_indices] = self.alpha3
                    loss[i] *= penalty_vector
            penalty_vector_cpu = penalty_vector.cpu().numpy()
            # if np.sum(penalty_vector_cpu == self.alpha3) > 0 and gt_whitelist_index[0] != 4:
            #     # print(gt_whitelist_index)
            #     # print("Penalized greatly: ", np.sum(penalty_vector_cpu != self.alpha3))
            #     print(xs_pos[i][penalty_vector_cpu != self.alpha3])
            #     # print(np.array(self.class_list)[y_np[i].cpu().numpy() == 1])
            #     # print(np.array(self.class_list)[penalty_vector_cpu != self.alpha3])
            #     print("++++++++++++")

        if self.gamma_neg > 0 or self.gamma_pos > 0 :
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()

class AsymmetricLossCustomPriorityLarge(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True,W=None, W_human=None, mapping_dict=None, limit=10, asymm=False, alpha1=None, alpha2=None, alpha3=None, penalize_other=False, alpha_other=None, weight=False):
        super(AsymmetricLossCustomPriorityLarge, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.W = W
        self.W_human = W_human
        state = torch.load("/data/yuhanl/Open_ImagesV6_TRresNet_L_448.pth", map_location='cpu')
        self.class_list = list(state['idx_to_class'].values())
        self.class_list = [i.replace("'", "").replace("\"", "") for i in self.class_list]
        recycle_labels = ['plastic', 'glass', 'paper', 'cardboard', 'metal', 'tin', 'carton']
        donate_labels = ['clothes', 'clothing', 'shirt', 'pants', 'jacket', 'footwear', 'shoe']
        compost_labels = ['food','snack','compost']
        self.asymm = asymm
        self.recycle_w_gt_ind = []
        for label in recycle_labels:
            if label not in mapping_dict:
                continue
            for value in mapping_dict[label][:limit]:
                self.recycle_w_gt_ind.append(self.class_list.index(value))

        self.donate_w_gt_ind = []
        for label in donate_labels:
            if label not in mapping_dict:
                continue
            for value in mapping_dict[label][:limit]:
                self.donate_w_gt_ind.append(self.class_list.index(value))
        self.donate_w_gt_ind = np.array(self.donate_w_gt_ind)
        self.recycle_w_gt_ind = np.array(self.recycle_w_gt_ind)

        self.compost_w_gt_ind = []
        for label in compost_labels:
            if label not in mapping_dict:
                continue
            for value in mapping_dict[label][:limit]:
                self.compost_w_gt_ind.append(self.class_list.index(value))
        self.compost_w_gt_ind = np.array(self.compost_w_gt_ind)

        self.mapping_dict = mapping_dict
        self.alpha1 = alpha1
        self.alpha2 = alpha2

        self.penalize_other = penalize_other
        self.alpha_other = alpha_other
        print("PRIORITY LOSS!!!")
        print("penalize_other: ", self.penalize_other)
        print("recycle index: ", self.recycle_w_gt_ind)
        print("donate index: ", self.donate_w_gt_ind)
        print("compost index: ", self.compost_w_gt_ind)
        print(self.alpha1)

        w_human_lower = [i.lower().split(" ")[0] for i in W_human]

        self.whitelist_indices = [self.compost_w_gt_ind, self.recycle_w_gt_ind, self.donate_w_gt_ind]
        self.whitelist_mapping = {}
        self.total_indices = np.array([True for i in range(len(self.class_list))])
        self.wl_indices = np.array([False for i in range(len(self.class_list))])
        for item in self.whitelist_indices:
            self.total_indices[item] = False
            self.wl_indices[item] = True
        for i in range(len(W_human)):
            self.whitelist_mapping[self.class_list.index(W_human[i])] = find_whitelist(self.class_list.index(W_human[i]), self.compost_w_gt_ind, self.recycle_w_gt_ind, self.donate_w_gt_ind)
        print(self.whitelist_mapping)
        self.whitelist_indices_bool = []
        for item in self.whitelist_indices:
            indices =  np.array([False for i in range(len(self.class_list))])
            indices[item] = True
            self.whitelist_indices_bool.append(indices)
        self.priority_list = [1, 2, 3, 4]
        self.alpha3 = alpha3
        print("Large loss!!!")
    def find_w(self, j):
        if j in self.compost_w_gt_ind:
            return 1
        if j in self.recycle_w_gt_ind:
            return 2
        if j in self.donate_w_gt_ind:
            return 3
        return 4
    def convert_target(self, gt_whitelist_index, y_np):
        target = torch.zeros(4).cuda()
        target[gt_whitelist_index[0] - 1] = 1
        # if torch.sum(y_np[self.total_indices]) > 0:
        #     target[-1] = 1
        return target
    def convert_x_pos(self, xs_pos):
        x_pos_vec = torch.zeros(4).cuda()
        for i in range(3):
            x_pos_vec[i] = torch.max(xs_pos[self.whitelist_indices[i]])
        x_pos_vec[-1] = torch.max(xs_pos[self.total_indices])
        return x_pos_vec
    def find_gt_whitelist(self, j):
        gt_whitelist = []
        min_index = 99999
        if torch.sum(j[self.compost_w_gt_ind] ) > 0:
            gt_whitelist.append(1)
            min_index = min(min_index, self.priority_list.index(1))
        if torch.sum(j[self.recycle_w_gt_ind] ) > 0:
            gt_whitelist.append(2)
            min_index = min(min_index, self.priority_list.index(2))
        if torch.sum(j[self.donate_w_gt_ind] ) > 0:
            gt_whitelist.append(3)
            min_index = min(min_index, self.priority_list.index(3))
        if len(gt_whitelist) == 0:
            gt_whitelist.append(4)
        return gt_whitelist, min_index
    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        y_np = y
        loss = torch.zeros(y.shape).cuda()
        x_sigmoid_np = x_sigmoid
        for i in range(y.shape[0]):
            y_telda = np.zeros(y[i].shape)
            help_y_recycle = y_np[i][self.recycle_w_gt_ind]
            help_y_donate = y_np[i][self.donate_w_gt_ind]
            help_y_compost = y_np[i][self.compost_w_gt_ind]
            y_telda = y_np[i]
            active_ind = None # all the indices except the indices in the whitelist
            whitelist_ind = None
            arg_sort = torch.argsort(x_sigmoid[i], descending=True)
            loss[i] = y[i] * torch.log(xs_pos[i].clamp(min=self.eps)) + (1 - y[i]) * torch.log(xs_neg[i].clamp(min=self.eps))
            first_gt_whitelist_index = -1
            gt_whitelist_index, min_index = self.find_gt_whitelist(y_np[i])
            target_vector = self.convert_target(gt_whitelist_index, y_np[i])
            x_vector = self.convert_x_pos(xs_pos[i])
            # torch.max(xs_pos[i][mask] >= torch.sort(xs_pos[i], descending=True)[10])
            # target_vector = torch.unsqueeze(target_vector, 0)
            # x_vector = torch.unsqueeze(x_vector, 0)
            loss_ce = F.binary_cross_entropy(x_vector, target_vector)
            loss[i] = -loss[i] * self.alpha3 + loss_ce * (1 - self.alpha3)

        return loss.sum()
class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        print("Before hardtanh ", grad_output)
        print("After hardtanh ", F.hardtanh(grad_output, min_val=-1, max_val=1))
        return F.hardtanh(grad_output, min_val=-1, max_val=1)

class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, x):
        x = STEFunction.apply(x)
        return x





class AsymmetricLossCustomPriorityRank(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True,W=None, W_human=None, mapping_dict=None, limit=10, asymm=False, alpha1=None, alpha2=None, alpha3=None, penalize_other=False, alpha_other=None, weight=False, sigmoid = False):
        super(AsymmetricLossCustomPriorityRank, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.W = W
        self.W_human = W_human
        state = torch.load("/data/yuhanl/Open_ImagesV6_TRresNet_L_448.pth", map_location='cpu')
        self.class_list = list(state['idx_to_class'].values())
        self.class_list = [i.replace("'", "").replace("\"", "") for i in self.class_list]
        recycle_labels = ['plastic', 'glass', 'paper', 'cardboard', 'metal', 'tin', 'carton']
        donate_labels = ['clothes', 'clothing', 'shirt', 'pants', 'jacket', 'footwear', 'shoe']
        compost_labels = ['food','snack','compost']
        self.asymm = asymm
        self.recycle_w_gt_ind = []
        for label in recycle_labels:
            if label not in mapping_dict:
                continue
            for value in mapping_dict[label]:
                self.recycle_w_gt_ind.append(self.class_list.index(value))

        self.donate_w_gt_ind = []
        for label in donate_labels:
            if label not in mapping_dict:
                continue
            for value in mapping_dict[label]:
                self.donate_w_gt_ind.append(self.class_list.index(value))
        self.donate_w_gt_ind = np.array(self.donate_w_gt_ind)
        self.recycle_w_gt_ind = np.array(self.recycle_w_gt_ind)

        self.compost_w_gt_ind = []
        for label in compost_labels:
            if label not in mapping_dict:
                continue
            for value in mapping_dict[label]:
                self.compost_w_gt_ind.append(self.class_list.index(value))
        self.compost_w_gt_ind = np.array(self.compost_w_gt_ind)

        self.mapping_dict = mapping_dict
        self.alpha1 = alpha1
        self.alpha2 = alpha2

        self.penalize_other = penalize_other
        self.alpha_other = alpha_other
        print("PRIORITY LOSS!!!")
        print("penalize_other: ", self.penalize_other)
        print("recycle index: ", self.recycle_w_gt_ind)
        print("donate index: ", self.donate_w_gt_ind)
        print("compost index: ", self.compost_w_gt_ind)
        print(self.alpha1)

        w_human_lower = [i.lower().split(" ")[0] for i in W_human]

        self.whitelist_indices = [self.compost_w_gt_ind, self.recycle_w_gt_ind, self.donate_w_gt_ind]
        self.whitelist_mapping = {}
        self.total_indices = np.array([True for i in range(len(self.class_list))])
        self.wl_indices = np.array([False for i in range(len(self.class_list))])
        self.total_indices_raw = []
        for item in self.whitelist_indices:
            self.total_indices[item] = False
            self.wl_indices[item] = True
        for idx in range(len(self.total_indices)):
            if self.total_indices[idx]:
                self.total_indices_raw.append(idx)

        for i in range(len(W_human)):
            self.whitelist_mapping[self.class_list.index(W_human[i])] = find_whitelist(self.class_list.index(W_human[i]), self.compost_w_gt_ind, self.recycle_w_gt_ind, self.donate_w_gt_ind)
        print(self.whitelist_mapping)
        self.whitelist_indices_bool = []
        for item in self.whitelist_indices:
            indices =  np.array([False for i in range(len(self.class_list))])
            indices[item] = True
            self.whitelist_indices_bool.append(torch.Tensor(indices).cuda() > 0)
        self.priority_list = [1, 2, 3, 4]
        self.alpha3 = alpha3
        print("Large loss!!!")
        self.total_indices_cuda = torch.Tensor(self.total_indices).cuda() > 0
        self.wl_indices_cuda = torch.Tensor(self.wl_indices).cuda() > 0
        self.sigmoid = sigmoid
        print("Sigmoid: ", self.sigmoid)
        print(self.total_indices_cuda )
    def find_w(self, j):
        if j in self.compost_w_gt_ind:
            return 1
        if j in self.recycle_w_gt_ind:
            return 2
        if j in self.donate_w_gt_ind:
            return 3
        return 4
    def convert_target(self, gt_whitelist_index, y_np):
        target = torch.zeros(4).cuda()
        target[gt_whitelist_index[0] - 1] = 1
        # if torch.sum(y_np[self.total_indices]) > 0:
        #     target[-1] = 1
        return target
    def convert_x_pos(self, xs_pos):
        x_pos_vec = torch.zeros(4).cuda()
        for i in range(3):
            x_pos_vec[i] = xs_pos[self.whitelist_indices[i]].max(0)[0]
        x_pos_vec[-1] = xs_pos[self.total_indices].max(0)[0]
        return x_pos_vec
    def find_gt_whitelist(self, j):
        gt_whitelist = []
        min_index = 99999
        if torch.sum(j[self.compost_w_gt_ind] ) > 0:
            gt_whitelist.append(1)
            min_index = min(min_index, self.priority_list.index(1))
        if torch.sum(j[self.recycle_w_gt_ind] ) > 0:
            gt_whitelist.append(2)
            min_index = min(min_index, self.priority_list.index(2))
        if torch.sum(j[self.donate_w_gt_ind] ) > 0:
            gt_whitelist.append(3)
            min_index = min(min_index, self.priority_list.index(3))
        if len(gt_whitelist) == 0:
            gt_whitelist.append(4)
        return gt_whitelist, min_index
    def our_rank_loss(self, x1, x2):
        # ste = StraightThroughEstimator()
        # # print(f"In loss: {x2-x1}, output: {ste(x2-x1)}")
        # return ste(x2-x1)
        if x2 - x1 > 0:
            return  5 * torch.sigmoid(x2-x1)
        return torch.sigmoid(x2-x1)
    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        y_np = y
        loss = torch.zeros(y.shape).cuda()
        loss_rank_all = torch.zeros(y.shape[0]).cuda()
        x_sigmoid_np = x_sigmoid
        error = 0
        for i in range(y.shape[0]):
            y_telda = np.zeros(y[i].shape)
            help_y_recycle = y_np[i][self.recycle_w_gt_ind]
            help_y_donate = y_np[i][self.donate_w_gt_ind]
            help_y_compost = y_np[i][self.compost_w_gt_ind]
            y_telda = y_np[i]
            active_ind = None # all the indices except the indices in the whitelist
            whitelist_ind = None
            arg_sort = torch.argsort(x_sigmoid[i], descending=True)
            loss[i] = y[i] * torch.log(xs_pos[i].clamp(min=self.eps)) + (1 - y[i]) * torch.log(xs_neg[i].clamp(min=self.eps))
            first_gt_whitelist_index = -1
            gt_whitelist_index, min_index = self.find_gt_whitelist(y_np[i])
            whitelist_max_prob = self.convert_x_pos(xs_pos[i])
            loss_rank = 0
            rl = nn.MarginRankingLoss()
            x_sorted, _ = torch.sort(xs_pos[i], descending=True)


            if gt_whitelist_index[0] == 4:
                # pass
                #
                top_k_thres = max(x_sorted[10], 0.1)
                error += 1
                all_non_others_vec = xs_pos[i][self.wl_indices_cuda]
                # loss_rank += max(0, -(x_sorted[10] - torch.max(all_non_others_vec))) * self.alpha1
                if self.sigmoid:
                    loss_rank += self.our_rank_loss(top_k_thres, torch.max(all_non_others_vec))

                else:
                    loss_rank += max(0, -(top_k_thres - torch.max(all_non_others_vec)))
                loss[i] *= self.alpha1
                # loss_rank *= self.alpha1
            else:
                # loss_rank += self.our_rank_loss(whitelist_max_prob[gt_whitelist_index[0] - 1], x_sorted[10])
                top_k_thres = max(x_sorted[9], 0.1)
                non_others_non_gt_indices = torch.bitwise_xor(self.wl_indices_cuda, self.whitelist_indices_bool[gt_whitelist_index[0] - 1])
                # non_others_non_gt_indices_top_k = torch.bitwise_and(non_others_non_gt_indices, top_k_mask)
                # print(non_others_non_gt_indices_top_k)
                # print(torch.sum(non_others_non_gt_indices_top_k == True))
                non_others_non_gt = xs_pos[i][non_others_non_gt_indices ]
                if self.sigmoid:
                    loss_rank += self.our_rank_loss(torch.max(xs_pos[i][self.whitelist_indices_bool[gt_whitelist_index[0] - 1]]),  max(torch.max(non_others_non_gt), top_k_thres) )
                else:
                    loss_rank = max(0, -(torch.max(xs_pos[i][self.whitelist_indices_bool[gt_whitelist_index[0] - 1]]) - max(torch.max(non_others_non_gt),top_k_thres)) )
            # loss[i] = -loss[i] * self.alpha3 + loss_rank * (1 - self.alpha3)
            # print(loss_rank)
            loss_rank_all[i] = loss_rank
            # print("Loss: ", loss[i])
        # print("original loss: ", self.alpha3 * -loss.sum() )
        # print("Our loss: ", self.alpha2 * loss_rank_all.sum())
        # print("Error: ", error)
        # if self.gamma_neg > 0 or self.gamma_pos > 0 :
        #     if self.disable_torch_grad_focal_loss:
        #         torch.set_grad_enabled(False)
        #     pt0 = xs_pos * y
        #     pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
        #     pt = pt0 + pt1
        #     one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
        #     one_sided_w = torch.pow(1 - pt, one_sided_gamma)
        #     if self.disable_torch_grad_focal_loss:
        #         torch.set_grad_enabled(True)
        #     loss *= one_sided_w
        return self.alpha3 * -loss.sum() + self.alpha2 * loss_rank_all.sum()






# class AsymmetricLossCustomPriorityRankNew(nn.Module):
#     def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True,W=None, W_human=None, mapping_dict=None, limit=10, asymm=False, alpha1=None, alpha2=None, alpha3=None, penalize_other=False, alpha_other=None, weight=False, sigmoid = False):
#         super(AsymmetricLossCustomPriorityRankNew, self).__init__()

#         self.gamma_neg = gamma_neg
#         self.gamma_pos = gamma_pos
#         self.clip = clip
#         self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
#         self.eps = eps
#         self.W = W
#         self.W_human = W_human
#         state = torch.load("/data/yuhanl/Open_ImagesV6_TRresNet_L_448.pth", map_location='cpu')
#         self.class_list = list(state['idx_to_class'].values())
#         self.class_list = [i.replace("'", "").replace("\"", "") for i in self.class_list]
#         all_labels = list(mapping_dict.keys())
#         self.all_labels = all_labels
#         self.all_wl_indices = {}
#         self.asymm = asymm
#         for label in all_labels:
#             self.all_wl_indices[label] = []
#         for label in all_labels:
            
#             for value in mapping_dict[label]:
#                 self.all_wl_indices[label].append(self.class_list.index(value))


#         self.mapping_dict = mapping_dict
#         self.alpha1 = alpha1
#         self.alpha2 = alpha2

#         self.penalize_other = penalize_other
#         self.alpha_other = alpha_other
#         self.whitelist_indices = []
#         self.wl_index_list = []
#         for key in self.all_wl_indices:
#             self.whitelist_indices.append(np.array(self.all_wl_indices[key]))
#             self.wl_index_list.append(self.all_wl_indices[key])
#         self.whitelist_mapping = {}
#         self.total_indices = np.array([True for i in range(len(self.class_list))])
#         self.wl_indices = np.array([False for i in range(len(self.class_list))])
#         self.total_indices_raw = []
#         print(self.whitelist_indices)
#         for item in self.whitelist_indices:
#             self.total_indices[item] = False
#             self.wl_indices[item] = True
#         for idx in range(len(self.total_indices)):
#             if self.total_indices[idx]:
#                 self.total_indices_raw.append(idx)

#         self.whitelist_indices_bool = []
#         for item in self.whitelist_indices:
#             indices =  np.array([False for i in range(len(self.class_list))])
#             indices[item] = True
#             self.whitelist_indices_bool.append(torch.Tensor(indices).cuda() > 0)
#         self.priority_list = np.arange(len(all_labels))
#         self.alpha3 = alpha3
#         self.total_indices_cuda = torch.Tensor(self.total_indices).cuda() > 0
#         self.wl_indices_cuda = torch.Tensor(self.wl_indices).cuda() > 0
#         # self.wl_indices_cuda = torch.ones(len(self.class_list)).cuda()
#         self.sigmoid = sigmoid
#         print(f"alpha3: {self.alpha3}, alpha2: {self.alpha2} alpha1: {self.alpha1} alpha4: {self.alpha_other}")
#         print(self.wl_indices_cuda )

#     def find_gt_whitelist(self, j):
#         gt_whitelist = []
#         for key in self.all_wl_indices:
#             indices = np.array(self.all_wl_indices[key])
#             if torch.sum(j[indices]) > 0:
#                 gt_whitelist.append(self.all_labels.index(key))
#         if len(gt_whitelist) == 0:
#             gt_whitelist.append(len(self.all_labels))
#         return gt_whitelist
#     # def our_rank_loss(self, x1, x2, margin):

#     #     if x2 - x1 + margin > 0:
#     #         return  5 * torch.sigmoid(x2-x1 + margin)
#     #     return torch.sigmoid(x2-x1+ margin)
#     def our_rank_loss(self, x1, x2, margin):
#         if x2-x1+margin > 0:
#             return self.alpha2 * 1/(1+torch.exp(-self.alpha3 * (x2-x1+margin)))
#         return 1/(1+torch.exp(-self.alpha3 * (x2-x1+margin)))

#     def our_rank_loss_exp(self, x1, x2, margin):
#         x = x1-margin - x2
#         return -torch.log((1/(1+torch.exp(-10 * x))))
#     def find_incorrect(self, label, xs_pos):
#         incorrect_max = 0
#         for i in range(label ):
#             incorrect_max = max(incorrect_max, torch.max(xs_pos[self.whitelist_indices_bool[i]]))
#         return incorrect_max
#     def forward(self, x, y, y_neg):
#         """"
#         Parameters
#         ----------
#         x: input logits
#         y: targets (multi-label binarized vector)
#         """

#         # Calculating Probabilities
#         x_sigmoid = torch.sigmoid(x)
#         xs_pos = x_sigmoid
#         xs_neg = 1 - x_sigmoid

#         # Asymmetric Clipping
#         if self.clip is not None and self.clip > 0:
#             xs_neg = (xs_neg + self.clip).clamp(max=1)
#         y_np = y
#         loss = torch.zeros(y.shape).cuda()
#         loss_rank_all = torch.zeros(y.shape[0]).cuda()
#         x_sigmoid_np = x_sigmoid
#         error = 0
#         x1 = torch.zeros(y.shape[0]).cuda()
#         x2 = torch.zeros(y.shape[0]).cuda()
#         for i in range(y.shape[0]):

#             loss[i] = y[i] * torch.log(xs_pos[i].clamp(min=self.eps)) + (1 - y[i]) * torch.log(xs_neg[i].clamp(min=self.eps))
#             gt_whitelist_index = self.find_gt_whitelist(y_np[i])  
#             gt_whitelist_index_neg = self.find_gt_whitelist(y_neg[i])
#             # if len(gt_whitelist_index) == 0:
#             #     loss_rank = loss[i]          
#             x_sorted, _ = torch.sort(xs_pos[i], descending=True)
#             # top_k_thres = 0.5
#             top_k_thres = max(x_sorted[20], 0.5)
#             # breakpoint()
#             loss_rank = 0
#             if gt_whitelist_index[0] == len(self.all_labels):
#                 # pass
#                 #
#                 # top_k_thres = 0.5
#                 if self.sigmoid:
#                     top_k_thres = max(x_sorted[15], 0.5)
#                 else:
#                     top_k_thres = 0.5
#                 error += 1
#                 all_non_others_vec = xs_pos[i][self.wl_indices_cuda]
#                 # loss_rank += max(0, -(x_sorted[10] - torch.max(all_non_others_vec))) * self.alpha1

#                 loss_rank += self.our_rank_loss(top_k_thres, torch.max(all_non_others_vec), self.alpha1) 
                    
#             else:
#                 # loss_rank += self.our_rank_loss(whitelist_max_prob[gt_whitelist_index[0] - 1], x_sorted[10])
#                 top_k_thres = max(x_sorted[15], 0.5)
#                 # top_k_thres = 0.5
#                 # top_k_thres = max(top_k_thres, torch.max(all_others_vec))
#                 assert len(gt_whitelist_index) == 1
#                 gt_max_score = torch.max( xs_pos[i][self.whitelist_indices_bool[gt_whitelist_index[0]]])
#                 incorrect_max_score = self.find_incorrect(gt_whitelist_index[0], xs_pos[i])
                
#                 loss_rank += self.alpha_other * self.our_rank_loss(gt_max_score, top_k_thres, self.alpha1) 
#                 if incorrect_max_score > 0:
#                     loss_rank += self.alpha_other * self.our_rank_loss(top_k_thres, incorrect_max_score, self.alpha1) 
#             loss_rank_all[i] = loss_rank
#         return loss_rank_all.sum()





class AsymmetricLossCustomPriorityRankNew(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, alpha=0, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True,W=None, W_human=None, mapping_dict=None, limit=10, asymm=False, alpha1=None, alpha2=None, alpha3=None, penalize_other=False, alpha_other=None, weight=False, sigmoid = False):
        super(AsymmetricLossCustomPriorityRankNew, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.W = W
        self.W_human = W_human
        state = torch.load("/data/yuhanl/Open_ImagesV6_TRresNet_L_448.pth", map_location='cpu')
        self.class_list = list(state['idx_to_class'].values())
        self.class_list = [i.replace("'", "").replace("\"", "") for i in self.class_list]
        all_labels = list(mapping_dict.keys())
        self.all_labels = all_labels
        self.all_wl_indices = {}
        self.asymm = asymm
        for label in all_labels:
            self.all_wl_indices[label] = []
        for label in all_labels:
            
            for value in mapping_dict[label]:
                self.all_wl_indices[label].append(self.class_list.index(value))


        self.mapping_dict = mapping_dict
        self.alpha1 = alpha1
        self.alpha2 = alpha2

        self.penalize_other = penalize_other
        self.alpha_other = alpha_other
        self.whitelist_indices = []
        self.wl_index_list = []
        for key in self.all_wl_indices:
            self.whitelist_indices.append(np.array(self.all_wl_indices[key]))
            self.wl_index_list.append(self.all_wl_indices[key])
        self.whitelist_mapping = {}
        self.total_indices = np.array([True for i in range(len(self.class_list))])
        self.wl_indices = np.array([False for i in range(len(self.class_list))])
        self.total_indices_raw = []
        print(self.whitelist_indices)
        for item in self.whitelist_indices:
            self.total_indices[item] = False
            self.wl_indices[item] = True
        for idx in range(len(self.total_indices)):
            if self.total_indices[idx]:
                self.total_indices_raw.append(idx)

        self.whitelist_indices_bool = []
        for item in self.whitelist_indices:
            indices =  np.array([False for i in range(len(self.class_list))])
            indices[item] = True
            self.whitelist_indices_bool.append(torch.Tensor(indices).cuda() > 0)
        self.priority_list = np.arange(len(all_labels))
        self.alpha3 = alpha3
        self.total_indices_cuda = torch.Tensor(self.total_indices).cuda() > 0
        self.wl_indices_cuda = torch.Tensor(self.wl_indices).cuda() > 0
        # self.wl_indices_cuda = torch.ones(len(self.class_list)).cuda()
        self.sigmoid = sigmoid
        self.alpha = alpha
        print(f"alpha3: {self.alpha3}, alpha2: {self.alpha2} alpha1: {self.alpha1} alpha4: {self.alpha_other} alpha: {self.alpha}")
        print(self.wl_indices_cuda )

    def find_gt_whitelist(self, j):
        gt_whitelist = []
        for key in self.all_wl_indices:
            indices = np.array(self.all_wl_indices[key])
            if torch.sum(j[indices]) > 0:
                gt_whitelist.append(self.all_labels.index(key))
        if len(gt_whitelist) == 0:
            gt_whitelist.append(len(self.all_labels))
        return gt_whitelist
    # def our_rank_loss(self, x1, x2, margin):

    #     if x2 - x1 + margin > 0:
    #         return  5 * torch.sigmoid(x2-x1 + margin)
    #     return torch.sigmoid(x2-x1+ margin)
    def our_rank_loss(self, x1, x2, margin):
        if x2-x1+margin > 0:
            return self.alpha2 * 1/(1+torch.exp(-self.alpha3 * (x2-x1+margin)))
        return 1/(1+torch.exp(-self.alpha3 * (x2-x1+margin)))

    def our_rank_loss_exp(self, x1, x2, margin):
        x = x1-margin - x2
        return -torch.log((1/(1+torch.exp(-10 * x))))
    def find_incorrect(self, label, xs_pos, gt_whitelist_index_neg):
        incorrect_max = 0
        incorrect_neg =0
        for i in range(len(self.all_labels) ):
            if i != label:
                incorrect_max = max(incorrect_max, torch.max(xs_pos[self.whitelist_indices_bool[i]]))
            if i in gt_whitelist_index_neg:
                incorrect_neg = max(incorrect_neg, torch.max(xs_pos[self.whitelist_indices_bool[i]] ) )
        return incorrect_max, incorrect_neg
    # def find_incorrect(self, label, xs_pos):
    #     incorrect_max = 0
    #     for i in range(label ):
    #         incorrect_max = max(incorrect_max, torch.max(xs_pos[self.whitelist_indices_bool[i]]))
    #     return incorrect_max
    def forward(self, x, y, y_neg):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        y_np = y
        loss = torch.zeros(y.shape).cuda()
        loss_rank_all = torch.zeros(y.shape[0]).cuda()
        x_sigmoid_np = x_sigmoid
        error = 0
        x1 = torch.zeros(y.shape[0]).cuda()
        x2 = torch.zeros(y.shape[0]).cuda()
        for i in range(y.shape[0]):

            loss[i] = y[i] * torch.log(xs_pos[i].clamp(min=self.eps)) + (1 - y[i]) * torch.log(xs_neg[i].clamp(min=self.eps))
            gt_whitelist_index = self.find_gt_whitelist(y_np[i])  
            gt_whitelist_index_neg = self.find_gt_whitelist(y_neg[i])
            # if len(gt_whitelist_index) == 0:
            #     loss_rank = loss[i]          
            x_sorted, _ = torch.sort(xs_pos[i], descending=True)
            # top_k_thres = 0.5
            top_k_thres = max(x_sorted[20], 0.5)
            # breakpoint()
            loss_rank = 0
            if gt_whitelist_index[0] == len(self.all_labels):
                # pass
                
                assert len(gt_whitelist_index_neg) > 0
                top_k_thres = max(x_sorted[10], self.alpha_other)
                
                error += 1
                all_non_others_vec = xs_pos[i][self.wl_indices_cuda]
                # loss_rank += max(0, -(x_sorted[10] - torch.max(all_non_others_vec))) * self.alpha1

                loss_rank += (1-self.alpha) * self.our_rank_loss(top_k_thres, torch.max(all_non_others_vec), self.alpha1) 
                incorrect_neg = 0
                for k in range(len(self.all_labels)):
                    if k in gt_whitelist_index_neg:
                        incorrect_neg = max(incorrect_neg, torch.max(xs_pos[i][self.whitelist_indices_bool[k]]) )
                loss_rank += self.alpha * self.our_rank_loss(top_k_thres, incorrect_neg, self.alpha1) 
            else:
                # loss_rank += self.our_rank_loss(whitelist_max_prob[gt_whitelist_index[0] - 1], x_sorted[10])
                top_k_thres = max(x_sorted[10], self.alpha_other)
                # top_k_thres = 0.5
                # top_k_thres = max(top_k_thres, torch.max(all_others_vec))
                assert len(gt_whitelist_index) == 1
                gt_max_score = torch.max( xs_pos[i][self.whitelist_indices_bool[gt_whitelist_index[0]]])
                incorrect_max_score, incorrect_neg = self.find_incorrect(gt_whitelist_index[0], xs_pos[i], gt_whitelist_index_neg)
                
                loss_rank +=  self.our_rank_loss(gt_max_score, top_k_thres, self.alpha1) 
                if incorrect_max_score > 0:
                    loss_rank += (1-self.alpha) * self.our_rank_loss(top_k_thres, incorrect_max_score, self.alpha1) 
                if incorrect_neg > 0:
                    loss_rank += self.alpha * self.our_rank_loss(top_k_thres, incorrect_neg, self.alpha1) 
                else:
                    loss_rank += self.alpha * self.our_rank_loss(top_k_thres, incorrect_max_score, self.alpha1) 
            loss_rank_all[i] = loss_rank
        return loss_rank_all.sum()



