#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn

# custom implementation of CBRTiny https://arxiv.org/pdf/1902.07208.pdf
# ----------- Implementation details -----------------------
# CBR-Tiny has 5x5 conv filters: 
        # (conv64-bn-relu) maxpool 
        # (conv128-bn-relu) maxpool 
        # (conv256-bn-relu) maxpool 
        # (conv512-bn-relu) maxpool, 
        # global avgpool, 
        # classification.
# Each maxpool has spatial window (3x3) and stride (2x2).
# Convolutions all have stride 1 
               
class CBRTinyBlock(nn.Module):
    def __init__(self, channel_in, channel_out, conv_kwargs, pool_kwargs):
        super(CBRTinyBlock, self).__init__()
        
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.conv_kwargs = conv_kwargs
        self.pool_kwargs = pool_kwargs 
        
        self.conv   = nn.Conv2d(channel_in, channel_out, **self.conv_kwargs)
        self.bn = nn.BatchNorm2d(channel_out)
        self.relu  = nn.ReLU()
        
        self.maxpool = nn.MaxPool2d(**pool_kwargs)

    def forward(self, x):
        out = self.maxpool(self.relu(self.bn(self.conv(x))))
        return out

class CBRTiny(nn.Module):
    def __init__(self,num_classes, channel_in=1, conv_kwargs=None,pool_kwargs=None):
        super(CBRTiny, self).__init__()
        self.num_classes = num_classes
        self.channel_in = channel_in

        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 5}
        if pool_kwargs is None:
            pool_kwargs = {'kernel_size': 3, 'stride': 2}
        
        self.conv_kwargs = conv_kwargs
        self.pool_kwargs = pool_kwargs
        
            
        # CBR Tiny Blocks (conv bn relu maxpool)
        self.b1 = CBRTinyBlock(self.channel_in,64,self.conv_kwargs, self.pool_kwargs)
        self.b2 = CBRTinyBlock(64,128,self.conv_kwargs, self.pool_kwargs)
        self.b3 = CBRTinyBlock(128,256,self.conv_kwargs, self.pool_kwargs)
        self.b4 = CBRTinyBlock(256,512,self.conv_kwargs, self.pool_kwargs)
        
        # global average pooling 
        self.avgpool = nn.AdaptiveAvgPool2d(1) 
        
        # classification layer 
        self.classification = nn.Linear(512,self.num_classes) 

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.avgpool(x).squeeze(dim=-1).squeeze(dim=-1)
        return self.classification(x)

        
       
