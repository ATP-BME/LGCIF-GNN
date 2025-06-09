import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, MaxPool1d, Linear, GRU
import math
import os
from torch.autograd import Function

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).cuda()
    return -torch.autograd.Variable(torch.log(-torch.log(U + eps) + eps))


def gumbel_softmax_sample(logits, temperature, eps=1e-10):
    sample = sample_gumbel(logits.size(), eps=eps)
    y = logits + sample
    return F.softmax(y / temperature, dim=-1)

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None



class GruKRegion(nn.Module):

    def __init__(self, kernel_size=8, layers=4, out_size=8, dropout=0.5):
        super().__init__()
        self.gru = GRU(kernel_size, kernel_size, layers,
                       bidirectional=True, batch_first=True)

        self.kernel_size = kernel_size

        self.linear = nn.Sequential(
            nn.Dropout(dropout),
            Linear(kernel_size*2, kernel_size),
            nn.LeakyReLU(negative_slope=0.2),
            # nn.Tanh(),
            Linear(kernel_size, out_size)
        )

    def forward(self, raw):

        b, k, d = raw.shape # batch,nROI,timeseries

        x = raw.view((b*k, -1, self.kernel_size))

        x, h = self.gru(x)

        x = x[:, -1, :]

        x = x.view((b, k, -1))

        x = self.linear(x)
        return x



class Embed2GraphByProduct(nn.Module):

    def __init__(self, input_dim, roi_num=264):
        super().__init__()

    def forward(self, x):

        m = torch.einsum('ijk,ipk->ijp', x, x)
        m = torch.unsqueeze(m, -1) 

        return m


class Embed2GraphByLinear(nn.Module):

    def __init__(self, input_dim, roi_num=360):
        super().__init__()

        self.fc_out = nn.Linear(input_dim * 2, input_dim)
        self.fc_cat = nn.Linear(input_dim, 1)

        def encode_onehot(labels):
            classes = set(labels)
            classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                            enumerate(classes)}
            labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                     dtype=np.int32)
            return labels_onehot

        off_diag = np.ones([roi_num, roi_num])
        rel_rec = np.array(encode_onehot(
            np.where(off_diag)[0]), dtype=np.float32)
        rel_send = np.array(encode_onehot(
            np.where(off_diag)[1]), dtype=np.float32)
        self.rel_rec = torch.FloatTensor(rel_rec).cuda()
        self.rel_send = torch.FloatTensor(rel_send).cuda()

    def forward(self, x):

        batch_sz, region_num, _ = x.shape
        receivers = torch.matmul(self.rel_rec, x)

        senders = torch.matmul(self.rel_send, x)
        x = torch.cat([senders, receivers], dim=2)
        x = torch.relu(self.fc_out(x))
        x = self.fc_cat(x)

        x = torch.relu(x)

        m = torch.reshape(
            x, (batch_sz, region_num, region_num, -1))
        return m



class GNNPredictor(nn.Module):

    def __init__(self, node_input_dim, site_num,roi_num=360,embed_dim=32):
        super().__init__()
        inner_dim = roi_num
        self.roi_num = roi_num
        self.gcn = nn.Sequential(
            nn.Linear(node_input_dim, inner_dim),
            nn.LeakyReLU(negative_slope=0.2),
            # nn.Tanh(),
            Linear(inner_dim, inner_dim)
        )
        self.bn1 = torch.nn.BatchNorm1d(inner_dim)

        self.gcn1 = nn.Sequential(
            nn.Linear(inner_dim, inner_dim),
            nn.LeakyReLU(negative_slope=0.2),
            # nn.Tanh(),
        )
        self.bn2 = torch.nn.BatchNorm1d(inner_dim)
        self.gcn2 = nn.Sequential(
            nn.Linear(inner_dim, 64),
            nn.LeakyReLU(negative_slope=0.2),
            # nn.Tanh(),
            nn.Linear(64, embed_dim), 
            nn.LeakyReLU(negative_slope=0.2),
            # nn.Tanh(),
        )
        self.bn3 = torch.nn.BatchNorm1d(inner_dim)
        self.attention = SelfAttention(roi_num,dim_q=32,dim_v=1)

        self.fcn = nn.Sequential(
            nn.Linear(8*roi_num, 256), 
            nn.LeakyReLU(negative_slope=0.2),
            # nn.Tanh(),
            nn.Linear(256, 32),
            nn.LeakyReLU(negative_slope=0.2),
            # nn.Tanh(),
            nn.Linear(32, 2)
        )
        
        self.site_d = nn.Sequential(
            nn.Linear(8*roi_num, 128), 
            nn.LeakyReLU(negative_slope=0.2),
            # nn.Tanh(),
            nn.Linear(128, 32),
            nn.LeakyReLU(negative_slope=0.2),
            # nn.Tanh(),
            nn.Linear(32, site_num)
        )


    def forward(self, m, node_feature,sub_names=None): # A, pearson correlaton matrix
        bz = m.shape[0]

        x = torch.einsum('ijk,ijp->ijp', m, node_feature) # j=k

        x = self.gcn(x)

        x = x.reshape((bz*self.roi_num, -1))
        x = self.bn1(x)
        x = x.reshape((bz, self.roi_num, -1))

        x = torch.einsum('ijk,ijp->ijp', m, x)

        x = self.gcn1(x)

        x = x.reshape((bz*self.roi_num, -1))
        x = self.bn2(x)
        x = x.reshape((bz, self.roi_num, -1))

        x = torch.einsum('ijk,ijp->ijp', m, x)

        x = self.gcn2(x)

        x = self.bn3(x) # [bs,nroi,embed dim]

        attention_score = F.softmax(torch.sum(m,dim=1),dim=1) * m.shape[-1]
        
        # self-attention
        x = attention_score.reshape(bz,-1,1)*x

        x = x.view(bz,-1)
        alpha = 0.5
        reverse_x = ReverseLayerF.apply(x, alpha)
        return self.fcn(x),self.site_d(reverse_x),x,attention_score


class SelfAttention(nn.Module):
    def __init__(self, input_dim, dim_q, dim_v):
        super(SelfAttention, self).__init__()
        # dim_q = dim_k
        self.dim_q, self.dim_k, self.dim_v = dim_q, dim_q, dim_v

        self.Q = nn.Linear(input_dim, dim_q)
        self.K = nn.Linear(input_dim, dim_q)
        self.V = nn.Linear(input_dim, dim_v)
        self._norm_fact = 1 / math.sqrt(self.dim_k)

    def forward(self, x):
        # Q: [batch_size,seq_len,dim_q]
        # K: [batch_size,seq_len,dim_k]
        # V: [batch_size,seq_len,dim_v]
        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)

        # print(f'x.shape:{x.shape} , Q.shape:{Q.shape} , K.shape: {K.shape} , V.shape:{V.shape}')

        attention = nn.Softmax(dim=-1)(
            torch.matmul(Q, K.permute(0, 2, 1)))  # Q * K.T() # batch_size * seq_len * seq_len

        attention = torch.matmul(attention, V).reshape(x.shape[0], x.shape[1],
                                                       -1)  # Q * K.T() * V # batch_size * seq_len * dim_v

        return attention

class LocalNet(nn.Module):

    def __init__(self, model_config, site_num,roi_num=360, node_feature_dim=360, time_series=512,embed_dim=32):
        super().__init__()
        self.graph_generation = model_config['graph_generation']
        if model_config['extractor_type'] == 'gru':
            self.extract = GruKRegion(
                out_size=model_config['embedding_size'], kernel_size=model_config['window_size'],
                layers=model_config['num_gru_layers'],dropout=model_config['dropout'])
            print('GRU dropout:',model_config['dropout'])
        if self.graph_generation == "linear":
            self.emb2graph = Embed2GraphByLinear(
                model_config['embedding_size'], roi_num=roi_num)
        elif self.graph_generation == "product":
            self.emb2graph = Embed2GraphByProduct(
                model_config['embedding_size'], roi_num=roi_num)

        self.predictor = GNNPredictor(node_feature_dim,site_num=site_num, roi_num=roi_num,embed_dim=embed_dim)

    def forward(self, t, nodes,sub_names=None):
        x = self.extract(t) # x -> h
        # x = F.softmax(x, dim=-1)
        m = self.emb2graph(x) # A = h * h^T

        m = m[:, :, :, 0] # 前面插入了一个维度，这里取出来还是三个维度 [batch size,node num,node num]

        bz, _, _ = m.shape

        edge_variance = torch.mean(torch.var(m.reshape((bz, -1)), dim=1))

        local_output,local_site_output,local_embedding,attention_score = self.predictor(m, nodes,sub_names)

        return local_output,local_site_output,local_embedding,attention_score, m, edge_variance


        
