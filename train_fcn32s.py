import os
import os.path as osp
import torchvision.models as models
import torch
import torch.nn as nn
import fcn

configurations = {
    # same configuration as original fcn32s
    1: dict(
	max_iteration = 10000,
	lr = 1.0e-10,
	momentum = 0.99,
	weight_decay = 0.0005,
	interval_validate = 4000,
    )
}

def get_parameters(model, bias):
    std_module = (	
        nn.ReLU,
        nn.sequential,
        nn.Maxpool2d,
        nn.Dropout2d,
        fcn.models.FCN32s,
        fcn.models.FCN16s,
        fcn.models.FCN8s
    )
    
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if bias:
                yield m.bias
            else:
                yield m.weight
        elif isinstance(m, nn.ConvTranspose2d):
            # weight is frozen because it is just a bilinear upsampling
            if bias:
                assert m.bias is None
        elif isinstance(m, std_module):
            continue
        else:
            raise ValueError('Unexpected module: %s' % str(m))

def main():
    model = fcn.models.FCN32s(n_class = 2)
    start_epoch = 0
    start_iteration = 0
    vgg16 = models.vgg16(pretrained = True)
    model.copy_params_from_vgg16(vgg16)
    cfg = configurations
    optim = torch.optim.SGD(
        [
            {'params': get_parameters(model, bias=False)},
            {'params': get_parameters(model, bias=True),
             'lr': cfg['lr'] * 2, 'weight_decay': 0},
        ],
        lr=cfg['lr'],
        momentum=cfg['momentum'],
        weight_decay=cfg['weight_decay'])



