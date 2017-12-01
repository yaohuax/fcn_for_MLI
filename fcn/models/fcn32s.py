import os.path as osp
from torch.autograd import Variable
import torchvision.models as models
import numpy as np
import torch
#from torch import optim
import torch.nn as nn
from PIL import Image
import torchvision.transforms
import torch.nn.functional as F

#configurations = {
#    # same configuration as original fcn32s
#    1: dict(
#	max_iteration = 10000,
#	lr = 1.0e-10,
#	momentum = 0.99,
#	weight_decay = 0.0005,
#	interval_validate = 4000,
#    )
#}
#
#def get_parameters(model, bias):
#    std_module = (	
#        nn.ReLU,
#        nn.Sequential,
#        nn.MaxPool2d,
#        nn.Dropout2d,
##        fcn.models.FCN32s,
##        fcn.models.FCN16s,
##        fcn.models.FCN8s
#    )
#    
#    for m in model.modules():
#        if isinstance(m, nn.Conv2d):
#            if bias:
#                yield m.bias
#            else:
#                yield m.weight
#        elif isinstance(m, nn.ConvTranspose2d):
#            # weight is frozen because it is just a bilinear upsampling
#            if bias:
#                assert m.bias is None
#        elif isinstance(m, std_module):
#            continue
#        else:
#            raise ValueError('Unexpected module: %s' % str(m))
#
#def cross_entropy(input, target, weight=None, size_average=True):
## input: (n, c, h, w), target: (n, h, w)
#    n, c, h, w = input.size()
#    # log_p: (n, c, h, w)
#    log_p = F.log_softmax(input)
#    # log_p: (n*h*w, c)
#    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
#    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
#    log_p = log_p.view(-1, c)
#    # target: (n*h*w,)
#    mask = target >= 0
#    target = target[mask]
#    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
#    if size_average:
#        loss /= mask.data.sum()
#    return loss
#
#def transform(img, lbl):
#    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
#    img = img[:, :, ::-1]  # RGB -> BGR
#    img = img.astype(np.float64)
#    img -= mean_bgr
#    img = img.transpose(2, 0, 1)
#    img = torch.from_numpy(img).float()
#    lbl = torch.from_numpy(lbl).long()
#    return img, lbl
#
#def untransform(img):
#        #img = img.numpy()
#        img = img.transpose(1, 2, 0)
#        img = img.astype(np.uint8)
#        img = img[:, :, ::-1]
#        return img

def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """
    The factor of upsampling is equal to the stride of transposed convolution
    Make a 2D bilinear kernal for upsampling
    """
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:	
	center = factor - 1
    else:
	center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()

class FCN32s(nn.Module):
    """
    This Convnet returns a tensor with size[n, n_class, H, W]
    the input size is [n, 3, H, W]
    """
    def __init__(self, n_class=21):		
        super(FCN32s, self).__init__()
        #conv1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding = 100)
        self.relu1_1 = nn.ReLU(inplace = True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding = 1)
        self.relu1_2 = nn.ReLU(inplace = True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        #conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding = 1)
        self.relu2_1 = nn.ReLU(inplace = True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding = 1)
        self.relu2_2 = nn.ReLU(inplace = True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        #conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()
        self.score_fr = nn.Conv2d(4096, n_class, 1)
        self.upscore = nn.ConvTranspose2d(n_class, n_class, 64, stride=32, bias=False)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        h = self.score_fr(h)

        h = self.upscore(h)
        #print h.data.shape
        h = h[:, :, 19:19 + x.size()[2], 19:19 + x.size()[3]].contiguous() 

        return h

    def copy_params_from_vgg16(self, vgg16):
        features = [
            self.conv1_1, self.relu1_1,
            self.conv1_2, self.relu1_2,
            self.pool1,
            self.conv2_1, self.relu2_1,
            self.conv2_2, self.relu2_2,
            self.pool2,
            self.conv3_1, self.relu3_1,
            self.conv3_2, self.relu3_2,
            self.conv3_3, self.relu3_3,
            self.pool3,
            self.conv4_1, self.relu4_1,
            self.conv4_2, self.relu4_2,
            self.conv4_3, self.relu4_3,
            self.pool4,
            self.conv5_1, self.relu5_1,
            self.conv5_2, self.relu5_2,
            self.conv5_3, self.relu5_3,
            self.pool5,
        ]
        for l1, l2 in zip(vgg16.features, features):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
        for i, name in zip([0, 3], ['fc6', 'fc7']):
            l1 = vgg16.classifier[i]
            l2 = getattr(self, name)
            l2.weight.data = l1.weight.data.view(l2.weight.size())
            l2.bias.data = l1.bias.data.view(l2.bias.size())

##def main():
#path = '/Users/yaohuaxu/Desktop/VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg'
#file = '/Users/yaohuaxu/Desktop/fcn32s_from_caffe.pth'
#label= '/Users/yaohuaxu/Desktop/VOCdevkit/VOC2012/SegmentationClass/2007_000032.png'
#img = Image.open(path)
#img = np.array(img)
#lbl = Image.open(label)
#lbl = np.array(lbl, dtype=np.int32)
#lbl[lbl == 255] = -1
#img,lbl= transform(img,lbl)
##print img.shape
#
#img = img.view((1,3,281, 500))
##print img.shape
#img, lbl = Variable(img), Variable(lbl)
##print img.data.shape
#net = FCN32s()
##num = net.score_fr.in_channels
##print num
##vgg16 = models.vgg16(pretrained = True)
##net.copy_params_from_vgg16(vgg16)
#cfg = configurations
#for m in net.modules():
#    print m
##optim = torch.optim.SGD(
##    [
##        {'params': get_parameters(net, bias=False)},
##        {'params': get_parameters(net, bias=True),
##         'lr': cfg[1]['lr'], 'weight_decay': 0},
##    ],
##    lr=cfg[1]['lr'],
##    momentum=cfg[1]['momentum'],
##    weight_decay=cfg[1]['weight_decay'])
#
#net.load_state_dict(torch.load(file))
##optimization = optim.Adam(net.parameters(), lr = 0.001)
#output = net.forward(img)
#loss = cross_entropy(output, lbl)
#if np.isnan(float(loss.data[0])):
#    raise ValueError('loss is nan while training')
#loss.backward()
#optimization.step()
#score = output.data.max(1)[1].cpu().numpy()
#score = untransform(score)
#print score.shape
#print output.data.shape
#
##if __name__ == '__main__':
##    main()
