import torch.nn as nn
import torch

def group_weight(module):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups


def create_optimizers(nets, param):
    (net_encoder, net_decoder, crit) = nets
    optimizer_encoder = torch.optim.SGD(
        group_weight(net_encoder),
        lr=param['lr_encode'],
        momentum=param['beta1'],
        weight_decay=param['weight_decay'])
    optimizer_decoder = torch.optim.SGD(
        group_weight(net_decoder),
        lr=param['lr_decode'],
        momentum=param['beta1'],
        weight_decay=param['weight_decay'])
    return (optimizer_encoder, optimizer_decoder)


def adjust_learning_rate(optimizers, cur_iter, param):
    if param['max_iters'] >= cur_iter:
        scale_running_lr = 1.0
    else:
        scale_running_lr = ((1. - float(cur_iter) / param['total_iters']) ** param['lr_pow'])
    param['running_lr_encoder'] = param['lr_encode'] * scale_running_lr
    param['running_lr_decoder'] = param['lr_decode'] * scale_running_lr

    (optimizer_encoder, optimizer_decoder) = optimizers
    for param_group in optimizer_encoder.param_groups:
        param_group['lr'] = param['running_lr_encoder']
    for param_group in optimizer_decoder.param_groups:
        param_group['lr'] = param['running_lr_decoder']

def adjust_learning_rate2(optimizer, cur_iter, param):
    if param['max_iters'] >= cur_iter:
        scale_running_lr = 1.0
    else:
        scale_running_lr = ((1. - float(cur_iter) / param['total_iters']) ** param['lr_pow'])
    param['running_lr'] = param['lr'] * scale_running_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = param['running_lr']