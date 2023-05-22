# -*- coding: utf-8 -*-
""" Created on Mon May  5 15:14:25 2023
 @author: Gorgen
  @Fuction： （1）“Dynamic Causal Explanation Based Diffusion-Variational Graph Neural Network for Spatio-temporal Forecasting”； #
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import BatchNorm2d, Conv2d, Parameter


from utils import ST_BLOCK_2

class DVGCN(nn.Module):
    def __init__(self,c_in,c_out,num_nodes,week,day,recent,K,Kt):
        super(DVGCN,self).__init__()
        tem_size=week+day+recent
        self.block1=ST_BLOCK_2(c_in,c_out,num_nodes,tem_size,K,Kt)
        self.block2=ST_BLOCK_2(c_out,c_out,num_nodes,tem_size,K,Kt)
        self.bn=BatchNorm2d(c_in,affine=False)
        
        self.conv1=Conv2d(c_out,1,kernel_size=(1, 1),padding=(0,0),
                          stride=(1,1), bias=True)
        self.conv2=Conv2d(c_out,1,kernel_size=(1, 1),padding=(0,0),
                          stride=(1,1), bias=True)
        self.conv3=Conv2d(c_out,1,kernel_size=(1, 1),padding=(0,0),
                          stride=(1,1), bias=True)
        self.conv4=Conv2d(c_out,1,kernel_size=(1, 2),padding=(0,0),
                          stride=(1,2), bias=True)
        
        self.h=Parameter(torch.zeros(num_nodes,num_nodes), requires_grad=True)
        nn.init.uniform_(self.h, a=0, b=0.0001)
        
    def forward(self,x_w,x_d,x_r,supports,adj_r):
        x_w=self.bn(x_w)
        x_d=self.bn(x_d)
        x_r=self.bn(x_r)
        x=torch.cat((x_w,x_d,x_r),-1)
        
        A=self.h+supports
        d=1/(torch.sum(A,-1)+0.0001)
        D=torch.diag_embed(d)
        A=torch.matmul(D,A)
        A1=F.dropout(A,0.8,self.training)
              
        x,_,_=self.block1(x,A1,adj_r)
        x,d_adj,t_adj=self.block2(x,A1,adj_r)
    
        x1=x[:,:,:,0:1]
        x2=x[:,:,:,1:2]
        x3=x[:,:,:,2:3]

        
        x1=self.conv1(x1).squeeze()
        x2=self.conv2(x2).squeeze()
        x3=self.conv3(x3).squeeze()

        x=x1+x2+x3
        return x,d_adj,A 
