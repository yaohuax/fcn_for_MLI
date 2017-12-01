import math
import os
import os.path as osp

import fcn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import tqdm


def cross_entropy(input, target, weight=None, size_average=True):
# input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    log_p = F.log_softmax(input)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss

class Trainer(object):
    def __init__(self, cuda, model, optimizer, train_loader, val_loader, max_iter, size_average = False):
        self.model = model
        self.cuda = cuda
        self.optim = optimizer
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        
#        self.out = out
#        if not osp.exist(self.out):
#            os.makedirs(self.out)

        self.size_average = size_average

        self.epoch = 0
        self.iteration = 0
        self.max_iter = max_iter
        self.best_mean_iu = 0
    
    def train_epoch(self):
        self.model.train()
        n_class = len(self.train_loader.dataset.class_names)

        for batch_idx, (data, target) in tqdm.tqdm(
                enumerate(self.train_loader), total=len(self.train_loader),
                desc='Train epoch=%d' % self.epoch, ncols=80, leave=False):
            iteration = batch_idx + self.epoch * len(self.train_loader)
            if self.iteration != 0 and (iteration - 1) != self.iteration:
                continue  # for resuming
            self.iteration = iteration

            assert self.model.training

            if self.cuda:
            	data, target = data.cuda(), target.cuda()

            """
            back_propagation
            """
            data, target = Variable(data), Variable(target)
            self.optim.zero_grad()
            score = self.model(data)

            loss = cross_entropy(score, target, size_average = self.size_average)

            #loss = loss / len(data)
            if np.isnan(np.float(loss.data[0])):
            	raise ValueError('loss is nan while training')
            loss.backward()
            self.optim.step()
            
            """
            measure accuracy by fcn.utils.label_accuracy_score
            """
            metrics = []
            lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = target.data.cpu().numpy()
            for lt, lp in zip(lbl_true, lbl_pred):
                acc, acc_cls, mean_iu, fwavacc = \
                    fcn.utils.label_accuracy_score(
                        [lt], [lp], n_class=n_class)
                metrics.append((acc, acc_cls, mean_iu, fwavacc))
            metrics = np.mean(metrics, axis=0)

            if self.iteration >= self.max_iter:
            	break
    
    def train(self):
        max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader)))
        for epoch in tqdm.trange(self.epoch, max_epoch,
                                 desc='Train', ncols=80):
            self.epoch = epoch
            self.train_epoch()
            if self.iteration >= self.max_iter:
                break
