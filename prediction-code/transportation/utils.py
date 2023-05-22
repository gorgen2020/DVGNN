# -*- coding: utf-8 -*-
""" Created on Mon May  5 15:14:25 2023
@author: Gorgen
@Fuction： （1）“Dynamic Causal Explanation Based Diffusion-Variational Graph Neural Network for Spatio-temporal Forecasting”；

# """

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, LayerNorm, BatchNorm1d
import numpy as np
"""
x-> [batch_num,in_channels,num_nodes,tem_size],
"""

###DVGCN
class T_cheby_conv_ds(nn.Module):
    '''
    x : [batch_size, feat_in, num_node ,tem_size] - input of all time step
    nSample : number of samples = batch_size
    nNode : number of node in graph
    tem_size: length of temporal feature
    c_in : number of input feature
    c_out : number of output feature
    adj : laplacian
    K : size of kernel(number of cheby coefficients)
    W : cheby_conv weight [K * feat_in, feat_out]
    '''
    def __init__(self,c_in,c_out,K,Kt):
        super(T_cheby_conv_ds,self).__init__()
        c_in_new=(K)*c_in
        self.conv1=Conv2d(c_in_new, c_out, kernel_size=(1, Kt),padding=(0,1),
                          stride=(1,1), bias=True)
        self.K=K
        
        
    def forward(self,x,adj):
        nSample, feat_in, nNode, length  = x.shape
        
        Ls = []
        L1 = adj
        L0 = torch.eye(nNode).repeat(nSample,1,1).cuda()
        Ls.append(L0)
        Ls.append(L1)
        for k in range(2, self.K):
            L2 = 2 *torch.matmul(adj, L1) - L0
            L0, L1 = L1, L2
            Ls.append(L2)

        Lap = torch.stack(Ls, 1)  # [B, K,nNode, nNode]
        #print(Lap)
        Lap = Lap.transpose(-1,-2)
        x = torch.einsum('bcnl,bknq->bckql', x, Lap).contiguous()
        x = x.view(nSample, -1, nNode, length)
        out = self.conv1(x)
        return out 
A=np.zeros((60,60))
for i in range(12):
    for j in range(12):
        A[i,j]=1
        A[i+12,j+12]=1
        A[i+24,j+24]=1
for i in range(24):
    for j in range(24):
        A[i+36,j+36]=1
B=(-1e13)*(1-A)
B=(torch.tensor(B)).type(torch.float32).cuda()
class TATT_1(nn.Module):
    def __init__(self,c_in,num_nodes,tem_size):
        super(TATT_1,self).__init__()
        self.conv1=Conv2d(c_in, 1, kernel_size=(1, 1),
                          stride=(1,1), bias=False)
        self.conv2=Conv2d(num_nodes, 1, kernel_size=(1, 1),
                          stride=(1,1), bias=False)
        self.w=nn.Parameter(torch.rand(num_nodes,c_in), requires_grad=True)
        nn.init.xavier_uniform_(self.w)
        self.b=nn.Parameter(torch.zeros(tem_size,tem_size), requires_grad=True)
        
        self.v=nn.Parameter(torch.rand(tem_size,tem_size), requires_grad=True)
        nn.init.xavier_uniform_(self.v)
        self.bn=BatchNorm1d(tem_size)
        
    def forward(self,seq):
        c1 = seq.permute(0,1,3,2)#b,c,n,l->b,c,l,n
        f1 = self.conv1(c1).squeeze()#b,l,n
        
        c2 = seq.permute(0,2,1,3)#b,c,n,l->b,n,c,l
        f2 = self.conv2(c2).squeeze()#b,c,n
     
        logits = torch.sigmoid(torch.matmul(torch.matmul(f1,self.w),f2)+self.b)
        logits = torch.matmul(self.v,logits)

        
        logits = logits.permute(0,2,1).contiguous()
        logits=self.bn(logits).permute(0,2,1).contiguous()
        coefs = torch.softmax(logits+B,-1)
        return coefs

    
    
class ST_BLOCK_2(nn.Module):
    def __init__(self,c_in,c_out,num_nodes,tem_size,K,Kt):
        super(ST_BLOCK_2,self).__init__()
        # self.device = torch.device("cuda:0")
        self.conv1=Conv2d(c_in, c_out, kernel_size=(1, 1),
                          stride=(1,1), bias=True)
        self.TATT_1=TATT_1(c_out,num_nodes,tem_size)
        self.dynamic_gcn=T_cheby_conv_ds(c_out,2*c_out,K,Kt)
        self.dynamic_gcn2 = T_cheby_conv_ds(c_out,c_out,K,Kt)
        self.K=K
        self.tem_size=tem_size
        self.time_conv=Conv2d(c_in, c_out, kernel_size=(1, Kt),padding=(0,1),
                          stride=(1,1), bias=True)
        self.c_out=c_out
        self.bn=LayerNorm([c_out,num_nodes,tem_size])

    def forward(self,x,supports,adj_r):
        shape = supports.shape

        x_input1=self.time_conv(x)

        if shape[0] == 207:
            x_input1=F.leaky_relu(x_input1)
        if shape[0] == 170:
            x_input1=F.leaky_relu(x_input1)


        x_input2 = self.dynamic_gcn2(x_input1,adj_r)
        x_1=self.dynamic_gcn(x_input2,adj_r)
        filter,gate=torch.split(x_1,[self.c_out,self.c_out],1)
        x_1=torch.sigmoid(gate)*F.leaky_relu(filter)
        # x_1=F.dropout(x_1,0.5,self.training)
        T_coef=self.TATT_1(x_1)
        T_coef=T_coef.transpose(-1,-2)
        x_1=torch.einsum('bcnl,blq->bcnq',x_1,T_coef)
        out=self.bn(F.leaky_relu(x_1)+x_input1)
        return out,adj_r,T_coef
