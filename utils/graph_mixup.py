import numpy as np
import random
import torch



def g_mixup(features,adj,inds,mixup_rate = 0.5):
    mix_lambda = np.random.beta(20,5,1)[0] # 比较均匀的mixup
    src = random.sample(list(inds.flatten()),int(inds.shape[0]*mixup_rate))
    tar = random.sample(list(inds.flatten()),int(inds.shape[0]*mixup_rate))
    
    features[src] = features[src] * mix_lambda + features[tar] * (1-mix_lambda)
    adj_mix = torch.zeros(adj.shape).to(torch.device(adj.device))
    adj_mix[src,tar] = min(mix_lambda,1-mix_lambda)
    adj_mix[tar,src] = min(mix_lambda,1-mix_lambda)
    adj_mix = adj.to_dense() + adj_mix
    adj_mix = adj_mix.to_sparse_coo()

    return features,adj_mix

# 2023-03-22 beta(16,16,1) mix_lambda -> beta(20,5,1) min(mix_lambda,1-mix_lambda)