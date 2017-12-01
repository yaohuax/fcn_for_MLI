#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 14:35:09 2017

@author: yaohuaxu
"""
from torch.utils import data
import collections
import os.path as osp
import argparse
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch.utils.data as data
import os
import scipy
from scipy import ndimage
import scipy.misc

def default_list_reader(fileList):
    imgList = []
    with open(fileList, 'r') as file:
        for line in file.readlines():
            imgshortList = []
            imgPath1, imgPath2= line.strip().split(' ')
            
            imgshortList.append(imgPath1)
            imgshortList.append(imgPath2)
            imgList.append(imgshortList)
    return imgList

def img_loader(path):
    img = Image.open(path)
    return img






class ImageList(data.Dataset):
    def __init__(self, fileList, transform=None, list_reader=default_list_reader, img_loader=img_loader):
        # self.root      = root
        self.imgList   = list_reader(fileList)
        # self._transform = transform
        self.img_loader = img_loader
        self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])

    def __getitem__(self, index):
        final = []
        [imgPath1, imgPath2] = self.imgList[index]
        # print "hello"
        img = self.img_loader(os.path.join("/home/lh/cv_ws/src/fcn_for_MLI/fcn/dataset/images", imgPath1))
        lbl = self.img_loader(os.path.join("/home/lh/cv_ws/src/fcn_for_MLI/fcn/dataset/labels", imgPath2))
        img, lbl = self.transform(img, lbl)
        return img, lbl

    def __len__(self):
        return len(self.imgList)

    def transform(self, img, lbl):
        img = np.array(img)
        lbl = np.array(lbl)
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl




def main():
    train_dataloader = torch.utils.data.DataLoader(
                        ImageList(fileList="/home/lh/cv_ws/src/fcn_for_MLI/train.txt", 
                        transform=transforms.Compose([ 
                                transforms.ToTensor(),            ])),
                        shuffle=False,
                        num_workers=8,
                        batch_size=1)
    for i, data in enumerate(train_dataloader,0):
        img, lbl= data
        print img, lbl

if __name__ == '__main__':
    main()