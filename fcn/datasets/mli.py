#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 14:35:09 2017

@author: yaohuaxu
"""
from torch.utils import data
import collections
import os.path as osp

class segBase(data.Dataset):
    #mean_bgr = p.array()
    def __init__(self, root, split = 'train', transform=False):
        self.root = root
        self.split = split
        self._transform = transform
        
        dataset_dir = osp.join(self.root, 'MLI/images')
        self.file = collections.defaultdict(list)
        
#        for split in ['train', 'val']:
    
    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        data_file = self.files[self.split][index]
        # load image
        img_file = data_file['img']
        img = PIL.Image.open(img_file)
        img = np.array(img, dtype=np.uint8)
        # load label
        lbl_file = data_file['lbl']
        lbl = PIL.Image.open(lbl_file)
        lbl = np.array(lbl, dtype=np.int32)
        lbl[lbl == 255] = -1
        if self._transform:
            return self.transform(img, lbl)
        else:
            return img, lbl
        
        
    
        
