import math, torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn.modules.module import Module
import torch.nn as nn
from opt import *

opt = OptInit().initialize()
device = opt.device

class general_GCN_layer(Module):
    def __init__(self):
        super(general_GCN_layer, self).__init__()

    @staticmethod
    def multiplication(A, B):
        if str(A.layout) == 'torch.sparse_coo':
            return torch.spmm(A, B)
        else:
            return torch.mm(A, B)

class snowball_layer(general_GCN_layer):
    '''
    realize one step of GCN to update embedding H
    '''
    def __init__(self, in_features, out_features):
        super(snowball_layer, self).__init__()
        self.in_features, self.out_features = in_features, out_features
        # self.weight, self.bias = Parameter(torch.FloatTensor(self.in_features, self.out_features).cuda()), Parameter(
        #     torch.FloatTensor(self.out_features).cuda())
        self.weight, self.bias = Parameter(torch.FloatTensor(self.in_features, self.out_features).to(device)), Parameter(
            torch.FloatTensor(self.out_features).to(device))
        self.reset_parameters()

    def reset_parameters(self):
        stdv_weight, stdv_bias = 1. / math.sqrt(self.weight.size(1)), 1. / math.sqrt(self.bias.size(0))
        torch.nn.init.uniform_(self.weight, -stdv_weight, stdv_weight)
        torch.nn.init.uniform_(self.bias, -stdv_bias, stdv_bias)

    def forward(self, input, adj, eye=False):
        XW = torch.mm(input, self.weight)
        if eye:
            return XW + self.bias # Hθ
        else:
            return self.multiplication(adj, XW) + self.bias # DADHθ+bias

class truncated_krylov_layer(general_GCN_layer):
    def __init__(self, in_features, n_blocks, out_features, LIST_A_EXP=None, LIST_A_EXP_X_CAT=None):
        super(truncated_krylov_layer, self).__init__()
        self.LIST_A_EXP = LIST_A_EXP
        self.LIST_A_EXP_X_CAT = LIST_A_EXP_X_CAT
        self.in_features, self.out_features, self.n_blocks = in_features, out_features, n_blocks
        self.shared_weight, self.output_bias = Parameter(
            torch.FloatTensor(self.in_features * self.n_blocks, self.out_features).cuda()), Parameter(
            torch.FloatTensor(self.out_features).cuda())
        self.reset_parameters()

    def reset_parameters(self):
        stdv_shared_weight, stdv_output_bias = 1. / math.sqrt(self.shared_weight.size(1)), 1. / math.sqrt(
            self.output_bias.size(0))
        torch.nn.init.uniform_(self.shared_weight, -stdv_shared_weight, stdv_shared_weight)
        torch.nn.init.uniform_(self.output_bias, -stdv_output_bias, stdv_output_bias)

    def forward(self, input, adj, eye=True):

        if self.n_blocks == 1:
            output = torch.mm(input, self.shared_weight)
            output = (output - output.mean(axis=0)) / output.std(axis=0)
        elif self.LIST_A_EXP_X_CAT is not None:
            output = torch.mm(self.LIST_A_EXP_X_CAT, self.shared_weight)
            output = (output - output.mean(axis=0)) / output.std(axis=0)
        elif self.LIST_A_EXP is not None:
            feature_output = []
            for i in range(self.n_blocks):
                AX = self.multiplication(self.LIST_A_EXP[i], input)
                feature_output.append(AX)
            output = torch.mm(torch.cat(feature_output, 1), self.shared_weight)
            output = (output - output.mean(axis=0)) / output.std(axis=0)
        if eye:
            return output + self.output_bias
        else:
            return self.multiplication(adj, output) + self.output_bias

class GraphConvolution2(nn.Module):

    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution2, self).__init__()
        self.variant = variant
        if self.variant:
            self.in_features = 2*in_features
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj , h0 , lamda, alpha, l):
        theta = math.log(lamda/l+1)
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi,h0],1)
            r = (1-alpha)*hi+alpha*h0
        else:
            support = (1-alpha)*hi+alpha*h0
            r = support
        output = theta*torch.mm(support, self.weight)+(1-theta)*r
        if self.residual:
            output = output+input
        return output

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        # support.cpu()
        output = torch.spmm(adj, support)
        # output = F.normalize(output)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class PairNorm(nn.Module):
    def __init__(self, mode='PN', scale=1):
        """
            mode:
              'None' : No normalization 
              'PN'   : Original version
              'PN-SI'  : Scale-Individually version
              'PN-SCS' : Scale-and-Center-Simultaneously version
           
            ('SCS'-mode is not in the paper but we found it works well in practice, 
              especially for GCN and GAT.)

            PairNorm is typically used after each graph convolution operation. 
        """
        assert mode in ['None', 'PN',  'PN-SI', 'PN-SCS']
        super(PairNorm, self).__init__()
        self.mode = mode
        self.scale = scale

        # Scale can be set based on origina data, and also the current feature lengths.
        # We leave the experiments to future. A good pool we used for choosing scale:
        # [0.1, 1, 10, 50, 100]
                
    def forward(self, x):
        if self.mode == 'None':
            return x
        
        col_mean = x.mean(dim=0)      
        if self.mode == 'PN':
            x = x - col_mean
            rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt() 
            x = self.scale * x / rownorm_mean

        if self.mode == 'PN-SI':
            x = x - col_mean
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual

        if self.mode == 'PN-SCS':
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual - col_mean

        return x