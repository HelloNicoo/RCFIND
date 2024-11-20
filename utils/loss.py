# -*- coding: utf-8 -*-
# @Time    : 2022/7/16 15:36
# @Author  : sylviazz
# @FileName: loss
import torch
from torch.nn import functional as F


def normalize_size(tensor):
    if len(tensor.size()) == 3:
        tensor = tensor.contiguous().view(-1, tensor.size(2))
    elif len(tensor.size()) == 2:
        tensor = tensor.contiguous().view(-1)

    return tensor


def calculate_entity_loss(pred_start, pred_end, gold_start, gold_end):
    pred_start = normalize_size(pred_start)
    pred_end = normalize_size(pred_end)
    gold_start = normalize_size(gold_start)
    gold_end = normalize_size(gold_end)

    weight = torch.tensor([1, 3]).float().cuda()

    loss_start = F.cross_entropy(pred_start, gold_start.long(), reduction='sum', ignore_index=-1)
    loss_end = F.cross_entropy(pred_end, gold_end.long(), reduction='sum', ignore_index=-1)

    return 0.5 * loss_start + 0.5 * loss_end
def caculate_rel_loss(rel_score, target):
    gamma = 2
    alpha = 1
    # onehot = torch.zeros(4, 3).cuda()
    # for i in range(len(target)):
    #     onehot[i][target[i]] = 1
    # # rel_score = F.softmax(rel_score, dim=-1)
    # log_p = F.log_softmax(rel_score)
    # pt = target * log_p
    # sub_pt = 1 - pt
    # fl = -alpha * (sub_pt) ** gamma * log_p
    # # loss = F.cross_entropy(rel_score, target.long(), reduction='sum', ignore_index=-1)
    # return fl
    # logpt = F.log_softmax(rel_score, dim=1)
    # pt = torch.exp(logpt)
    # logpt = (1 - pt) ** gamma * logpt
    # loss = F.nll_loss(logpt, target, ignore_index=-1)
    # return loss
    target = torch.Tensor(target).cuda()
    # rel_score = F.softmax(rel_score, dim=-1)
    loss = F.cross_entropy(rel_score, target.long(), reduction='sum', ignore_index=-1)
    return loss
