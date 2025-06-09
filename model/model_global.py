
import torch.nn as nn
import torch.nn.functional as F
from model.layers import PairNorm, snowball_layer
from model.GCN import snowball, graph_convolutional_network
import numpy as np
from torch.nn.parameter import Parameter
import torch
import networkx as nx
from torch.nn.modules.module import Module
import math
from opt import *
from utils.shift_robust_utils import *

opt = OptInit().initialize()
device = opt.device
# cudaid = "cuda:0"
# device = torch.device(cudaid)

def normalize( A, symmetric=True):

    A=A.cpu()
    d = A.sum(1)
    # d=torch.from_numpy(d)
    if symmetric:
        D = torch.diag(torch.pow(d, -0.5))
        return D.mm(A).mm(D)
    else:
        D = torch.diag(torch.pow(d, -1))
        return D.mm(A)


class snowball(graph_convolutional_network):
    def __init__(self, nfeat, nlayers, nhid, nclass, dropout_rate, activation):
        super(snowball, self).__init__(nfeat, nlayers, nhid, nclass, dropout_rate)
        self.activation = activation
        for k in range(nlayers):
            self.hidden.append(snowball_layer(k * nhid + nfeat, nhid)) # basic GCN layer
        self.out = snowball_layer(nlayers * nhid + nfeat, nclass) # basic GCN layer

        self.norm = PairNorm(opt.norm_mode, opt.norm_scale)

    def forward(self, x, adj):
        list_output_blocks = [] # list to save snowball embedding at different stage H1...Hn
        for layer, layer_num in zip(self.hidden, np.arange(self.nlayers)):
            if layer_num == 0:
                list_output_blocks.append(
                    F.dropout(self.activation(self.norm(layer(x, adj))), self.dropout_rate, training=self.training)) # pairnorm
            else:
                list_output_blocks.append(

                    F.dropout(self.activation(self.norm(layer(torch.cat([x] + list_output_blocks[0: layer_num], 1), adj))),
                              self.dropout_rate, training=self.training))
        output = self.out(torch.cat([x] + list_output_blocks, 1), adj, eye=False) # basic GCN


        output=F.normalize(output)
        return output


class Attention(nn.Module):
    '''
    input:
        concacted final graph embedding
    '''
    def __init__(self, in_size, hidden_size=2):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        # torch.backends.cudnn.enabled = False
        w = self.project(z)
        beta = torch.softmax(w, dim=1) # attention value
        # beta=1
        return (beta * z).sum(1), beta # a1Haal+a2Hho+a3Hc+a4Hp



class GlobalNet(nn.Module):
    # def __init__(self, nfeat, nclass, nhid1, nhid2, n, dropout):
    def __init__(self, nfeat, nhid,out, nclass,nhidlayer,dropout,baseblock,inputlayer,outputlayer,nbaselayer,activation,withbn,withloop,aggrmethod,mixmode,train_ind,test_ind):
        super(GlobalNet, self).__init__()

        self.train_ind = train_ind
        self.test_ind = test_ind
        # snowball_layer_num = 7 #default=9
        snowball_layer_num = opt.snowball_layer_num
        # print('snowball layer num:',snowball_layer_num)
        self.SGCN1 = snowball(nfeat, snowball_layer_num, nhid, out, dropout, nn.Tanh()) 
        self.SGCN2 = snowball(nfeat, snowball_layer_num, nhid, out, dropout, nn.Tanh()) 
        self.CGCN = snowball(nfeat, snowball_layer_num, nhid, out, dropout, nn.Tanh())


        self.dropout = dropout
        self.a = nn.Parameter(torch.zeros(size=(out, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.attention = Attention(out).to(device)
        self.tanh = nn.Tanh()

        self.MLP = nn.Sequential(
            nn.Linear(out, nclass),
            # nn.LogSoftmax(dim=1)
            nn.Softmax(dim=1)
        )

    # def forward(self, x, sadj, fadj,fadj2):
    def forward(self, x, padj, fadj, fadj2=None):
        emb1 = self.SGCN1(x, padj) # Special_GCN out1 
        emb2 = self.SGCN2(x, fadj) # Special_GCN out2 
        
        

        com1 = self.CGCN(x, padj)  # Common_GCN out1 
        com2 = self.CGCN(x, fadj)  # Common_GCN out2 
        
        
        
        Xcom = (com1 + com2 ) / 2


        emb = torch.stack([emb1,emb2,Xcom], dim=1)

        emb, att = self.attention(emb)
        output = self.MLP(emb)

        if opt.shift_robust:
            shift_loss = (self.shift_robust_output(emb1,self.train_ind,self.test_ind)\
                       + self.shift_robust_output(emb2,self.train_ind,self.test_ind)\
                       + self.shift_robust_output(emb,self.train_ind,self.test_ind)) / 4
        else:
            shift_loss = torch.zeros(1).to(torch.device(emb.device))

        return output,shift_loss,att, emb1, com1, com2, emb2
    
    def shift_robust_output(self,embedding,idx_src,idx_tar,alpha=1):
        if  len(idx_src) >= len(idx_tar):
            perm = torch.randperm(idx_src.shape[0])
            idx_src = idx_src[perm[:idx_tar.shape[0]]]
        else:
            perm = torch.randperm(idx_tar.shape[0])
            idx_tar = idx_tar[perm[:idx_src.shape[0]]]
        return alpha * cmd(embedding[idx_src,:],embedding[idx_tar,:])
    
