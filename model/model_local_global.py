import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear as Lin
import numpy as np
import yaml
from sklearn.metrics import accuracy_score

from utils.gcn_utils import normalize,normalize_torch
from utils.graph_mixup import g_mixup
from model.model_local import LocalNet
from model.model_global import GlobalNet

from opt import *
opt = OptInit().initialize()
device = opt.device

with open(opt.config_filename) as f:
        config = yaml.load(f, Loader=yaml.Loader)

class PAE(torch.nn.Module):
    def __init__(self, input_dim, dropout=0.2):
        super(PAE, self).__init__()
        hidden=128
        self.parser =nn.Sequential(
                nn.Linear(input_dim, hidden, bias=True),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(hidden),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden, bias=True),
                )
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-8)
        self.input_dim = input_dim
        self.model_init()
        self.relu = nn.ReLU(inplace=True)
        self.elu = nn.ReLU()

    def forward(self, x):
        x1 = x[:,0:self.input_dim]
        x2 = x[:,self.input_dim:]
        h1 = self.parser(x1) 
        h2 = self.parser(x2) 
        p = (self.cos(h1,h2) + 1)*0.5

        return p

    def model_init(self):
        for m in self.modules():
            if isinstance(m, Lin):  
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

class LGMF_GNN(torch.nn.Module):
    def __init__(self,nonimg,site_num,roi_num,local_fea_dim,global_fea_dim,timeseries_len,local_dataloader,train_HC_ind,train_MDD_ind,test_ind) -> None:
        super(LGMF_GNN,self).__init__()

        self.nonimg = nonimg
        self.site_num = site_num
        self.roi_num = roi_num
        self.local_fea_dim = local_fea_dim
        self.global_fea_dim = global_fea_dim
        self.time_series_len = timeseries_len
        
        self.local_dataloader = local_dataloader
        self.edge_dropout = opt.edropout
        self.train_HC_ind = train_HC_ind
        self.train_MDD_ind = train_MDD_ind
        self.train_ind = np.concatenate([train_HC_ind,train_MDD_ind])
        self.test_ind = np.array(test_ind)
        self._setup()
    
    def _setup(self):
        if opt.interp_grad:
            config['model']['dropout'] = 0
        self.local_gnn = LocalNet(config['model'], 
                                  site_num = self.site_num,
                                  roi_num=self.roi_num, 
                                  node_feature_dim=self.local_fea_dim, 
                                  time_series=self.time_series_len,
                                  embed_dim=config['model']['embedding_size'])
        self.global_gnn = GlobalNet(nfeat=self.global_fea_dim, 
                                  nhid=32, 
                                  out=16,
                                  nclass=2,
                                  nhidlayer=1,
                                  dropout=opt.dropout,
                                  baseblock="inceptiongcn",
                                  inputlayer="gcn",
                                  outputlayer="gcn",
                                  nbaselayer=6,
                                  activation=F.relu,
                                  withbn=False,
                                  withloop=False,
                                  aggrmethod="concat",
                                  mixmode=False,
                                  train_ind = self.train_ind,
                                  test_ind = self.test_ind)

        self.edge_net = PAE(2*self.nonimg.shape[1]//2,opt.dropout)

        self.local_loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        self.local_site_loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

    def eval(self):
        super(LGMF_GNN, self).eval()  
        self.local_gnn.eval()  
        self.global_gnn.eval()  


    def forward(self,dl,train_ind=None,n=None,feature_id=None,interp_dict = None):

        embeddings = []
        labels = []
        local_preds = []
        local_site_preds = []
        ids = []

        if not opt.train:
            local_m = []
            local_attention = []
        local_loss = 0
        local_site_loss = 0
        local_acc = 0
        local_site_acc = 0
        for data_in,pearson,label,sub_names in self.local_dataloader:
            label = label.long()
            
            data_in, pearson, label = data_in.to(device), pearson.to(device), label.to(device)
            
            local_output,local_site_output,embedding,attention_score,m,_ = self.local_gnn(data_in,pearson,sub_names)
            # local_loss += self.local_loss_fn(local_output,label)
            embeddings.append(embedding)
            
            labels.append(label)
            local_preds.append(local_output)
            local_site_preds.append(local_site_output)
            ids.append(sub_names)
            if not opt.train:
                local_attention.append(attention_score.squeeze())
                local_m.append(m)
        embeddings = torch.cat(tuple(embeddings))
        
        labels = torch.cat(tuple(labels))
        local_preds = torch.cat(tuple(local_preds))
        local_site_preds = torch.cat(tuple(local_site_preds))
        ids = torch.cat(tuple(ids))
        if not opt.train:
            local_m = torch.cat(tuple(local_m))
            local_attention = torch.cat(tuple(local_attention))
            np.save('HC_local_m.npy',local_m[self.train_HC_ind.reshape(-1)].detach().cpu().numpy())
            np.save('MDD_local_m.npy',local_m[self.train_MDD_ind.reshape(-1)].detach().cpu().numpy())
            np.save('HC_local_atten.npy',local_attention[self.train_HC_ind.reshape(-1)].detach().cpu().numpy())
            np.save('MDD_local_atten.npy',local_attention[self.train_MDD_ind.reshape(-1)].detach().cpu().numpy())
            np.save('test_local_atten.npy',local_attention[self.test_ind.reshape(-1)].detach().cpu().numpy())
            np.save('train_ind.npy',self.train_ind)
            np.save('test_ind.npy',self.test_ind)
            if interp_dict is not None:
                interp_dict['ids'].append(ids.detach().cpu().numpy())
                interp_dict['HC_local_m'].append(local_m[self.train_HC_ind.reshape(-1)].detach().cpu().numpy())
                interp_dict['MDD_local_m'].append(local_m[self.train_MDD_ind.reshape(-1)].detach().cpu().numpy())
                interp_dict['HC_local_atten'].append(local_attention[self.train_HC_ind.reshape(-1)].detach().cpu().numpy())
                interp_dict['MDD_local_atten'].append(local_attention[self.train_MDD_ind.reshape(-1)].detach().cpu().numpy())
                interp_dict['test_local_atten'].append(local_attention[self.test_ind.reshape(-1)].detach().cpu().numpy())


        if train_ind is not None:
            # disease cls
            local_loss = self.local_loss_fn(local_preds[train_ind],labels[train_ind])
            local_acc = accuracy_score(labels[train_ind].detach().cpu(),local_preds[train_ind].max(1)[1].detach().cpu())
            # site cls
            y_site = torch.from_numpy(dl.site).long().to(device)
            local_site_loss = self.local_site_loss_fn(local_site_preds[train_ind],y_site[train_ind].to(device))
            local_site_acc = accuracy_score(y_site[train_ind].detach().cpu(),local_site_preds[train_ind].max(1)[1].detach().cpu())
            
            # local_loss = local_loss - local_site_loss

        np.save('label.npy',{'labels':labels.detach().cpu().numpy(),'site_labels':dl.site})


        # normalize with norm
        embeddings = embeddings / torch.norm(embeddings, dim=-1, keepdim=True)
        
        


        edge_index, edge_input = dl.get_PAE_inputs(self.nonimg, embeddings) 
        # edge input: concat of noimg node feature of the nodes linked by the edge
        edge_input = (edge_input- edge_input.mean(axis=0)) / (edge_input.std(axis=0)+1e-6)
        # print(np.isnan(edge_input).any())
        edge_index = torch.tensor(edge_index, dtype=torch.long).to(opt.device)
        edge_input = torch.from_numpy(edge_input).to(opt.device)

        if self.edge_dropout > 0:
            if self.training and opt.train:
                one_mask = torch.ones([edge_input.shape[0],1]).to(device)
                self.drop_mask = F.dropout(one_mask, self.edge_dropout, True)
                self.bool_mask = torch.squeeze(self.drop_mask.type(torch.bool))
                edge_index = edge_index[:, self.bool_mask]
                edge_input = edge_input[self.bool_mask]

        
        edge_weight = torch.squeeze(self.edge_net(edge_input)) # W_i,j

        padj = torch.zeros([embeddings.shape[0], embeddings.shape[0]]).to(opt.device)

        for i in range(edge_index.shape[1]):
            padj[edge_index[0][i]][edge_index[1][i]] = edge_weight[i] 
            padj[edge_index[1][i]][edge_index[0][i]] = edge_weight[i]

        fadj1 = (torch.mm(embeddings,embeddings.T) +1 ) * 0.5
        
        if n is None and opt.n is None:
            k_num = range(8,12)
            k_num = np.random.choice(k_num, size=1)[0]
            

        else:
            if n is not None:
                k_num = n
            else:
                k_num = opt.n
        
        knn_fadj1 = torch.zeros_like(fadj1)
      
        knn_fadj1[torch.arange(len(fadj1)).unsqueeze(1),torch.topk(fadj1,k_num).indices]=1
      
        fadj1 = knn_fadj1

        fadj1 = prepare_adj(fadj1).to(opt.device)
        padj = prepare_adj(padj).to(opt.device)

        if not opt.train:
            np.save('fadj1.npy',fadj1.detach().cpu().numpy())
            np.save('padj.npy',padj.detach().cpu().numpy())

        if opt.mixup and self.training and opt.train:
            embeddings,fadj1 = g_mixup(embeddings,fadj1,self.train_HC_ind,mixup_rate = opt.mixup_rate)
            embeddings,fadj1 = g_mixup(embeddings,fadj1,self.train_MDD_ind,mixup_rate = opt.mixup_rate)

        
        node_logits,shift_loss, att, emb1, com1, com2, emb2 = self.global_gnn(embeddings, padj,fadj1)
        return node_logits, shift_loss, att, emb1, com1, com2, emb2,k_num,local_loss,local_site_loss,local_acc,local_site_acc,interp_dict


def prepare_adj(adj):

    nfadj = adj
    nfadj = nfadj + nfadj.T.multiply(nfadj.T > nfadj) - nfadj.multiply(nfadj.T > nfadj)
    nfadj = normalize_torch(torch.eye(nfadj.shape[0]).to(torch.device(nfadj.device)) + nfadj)
    return nfadj


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

class Sparsify_threshold(torch.nn.Module):
    """
    Sparsifyer
    """
    def __init__(self) -> None:

        super().__init__()
        self.threshold = torch.nn.parameter.Parameter(torch.full((1,), -10.0))
    def forward(self, adjacency_matrix: torch.Tensor) -> torch.Tensor:
        sparse_adjacency = torch.relu(adjacency_matrix - torch.sigmoid(self.threshold))
        return sparse_adjacency

