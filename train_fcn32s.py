import os
import os.path as osp
import torchvision.models as models
import torch
import torch.nn as nn
import trainer
# import fcn


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
    file = '/home/lh/cv_ws/src/fcn_for_MLI/fcn32s_from_caffe.pth'
    model = fcn.models.FCN32s()
    model.load_state_dict(torch.load(file))
    num = model.score_fc.out_channels
    model.score_fc = nn.Conv2d(num, 2, 1)
    model.upscore = nn.ConvTranspose2d(2,2,64, stride=32,bias=False)
    start_epoch = 0
    start_iteration = 0
    cfg = configurations
    optim = torch.optim.SGD(
        [
            {'params': get_parameters(model, bias=False)},
            {'params': get_parameters(model, bias=True),
             'lr': cfg[1]['lr'] * 2, 'weight_decay': 0},
        ],
        lr=cfg['lr'],
        momentum=cfg['momentum'],
        weight_decay=cfg['weight_decay'])




    train_dataloader = torch.utils.data.DataLoader(
                        ImageList(fileList="/home/lh/cv_ws/src/fcn_for_MLI/train.txt", 
                        transform=transforms.Compose([ 
                                transforms.ToTensor(),            ])),
                        shuffle=False,
                        num_workers=8,
                        batch_size=1)
    for i, data in enumerate(train_dataloader,0):
        # img, lbl= data
        trainer = Trainer(cuda=cuda, optimizer=optim, train_loader=train_dataloader, val_loader=train_dataloader,
                                max_iter = 500)
        trainer.train()