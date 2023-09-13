import os
import os.path as osp
import torch


def save_model(model, save_path):
    if osp.exists(osp.dirname(save_path)) == False:
        os.mkdir(osp.dirname(save_path))
    torch.save(model.state_dict(), save_path)


def get_param_size(model):
    param = 0
    for p in list(model.parameters()):
        n = 1
        for s in list(p.size()):
            n = n * s
        param += n
    return param