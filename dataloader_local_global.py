import data.Parser as Reader
import numpy as np
import torch
import torch.utils.data as utils
from utils.gcn_utils import preprocess_features
from sklearn.model_selection import StratifiedKFold,KFold
import yaml
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings("ignore")
from opt import *
opt = OptInit().initialize()
with open(opt.config_filename) as f:
    config = yaml.load(f, Loader=yaml.Loader)

class dataloader_lg():
    def __init__(self):
        self.pd_dict = {}
        self.num_classes = 2
        self.n_sub = 0

    def load_data(self, subject_IDs=None,connectivity='correlation', atlas1='aal',atlas2='cc200'):
        ''' 
        return: imaging features (raw), labels, non-image data
        '''
        if subject_IDs is None:
            subject_IDs = Reader.get_ids()
        # 标签
        labels = Reader.extract_label(sub=subject_IDs, score='缓解')  
        num_nodes = len(subject_IDs) # fot global gragh
        self.n_sub = num_nodes

        sites = Reader.extract_label(subject_IDs, score='site') 
        unique = np.unique(list(sites.values())).tolist()
        site_num = len(unique)
        ages = Reader.extract_label(sub=subject_IDs, score='age')
        genders = Reader.extract_label(sub=subject_IDs, score='sex')
        edus = Reader.extract_label(sub=subject_IDs, score='edu')
        durations = Reader.extract_label(sub=subject_IDs, score='Duration_day')
        QLESs = Reader.extract_label(sub=subject_IDs, score='QLES')
        YMSRs = Reader.extract_label(sub=subject_IDs, score='YMSR')
        HAMDs = Reader.extract_label(sub=subject_IDs, score='HAMD')


        

        # dsms=Reader.get_subject_score(subject_IDs,score='DSM_IV_TR') # 
        # protocols=Reader.get_subject_score(subject_IDs,score='protocol') 
        # hands=Reader.get_subject_score(subject_IDs,score='hand') #
        unique_labels = np.unique(list(labels.values())).tolist()
        unique_labels.sort()
        print('unique labels:',unique_labels)
        print('unique sites:',unique)

        y_onehot = np.zeros([num_nodes, self.num_classes])
        y = np.zeros([num_nodes])
        site = np.zeros([num_nodes], dtype=np.int32)
        edu = np.zeros([num_nodes], dtype=np.int32)
        age = np.zeros([num_nodes], dtype=np.float32)
        gender = np.zeros([num_nodes], dtype=np.int32)
        duration = np.zeros([num_nodes], dtype=np.int32)

        # dsm=np.zeros([num_nodes],dtype=np.int)
        qles = np.zeros([num_nodes,16],dtype=np.int32)
        ymsr = np.zeros([num_nodes,11],dtype=np.int32)
        hamd = np.zeros([num_nodes,17],dtype=np.int32)



        for i in range(num_nodes):
            y_onehot[i, unique_labels.index(labels[subject_IDs[i]])] = 1  # 0: [1,0]   1: [0,1]
            y[i] = unique_labels.index(labels[subject_IDs[i]])

            site[i] = unique.index(sites[subject_IDs[i]])  
            age[i] = float(ages[subject_IDs[i]])
            edu[i] = float(edus[subject_IDs[i]])
            gender[i] = genders[subject_IDs[i]]
            duration[i] = durations[subject_IDs[i]]

            qles[i] = QLESs[subject_IDs[i]]
            ymsr[i] = YMSRs[subject_IDs[i]]
            hamd[i] = HAMDs[subject_IDs[i]]
            # dsm[i] = dsms[subject_IDs[i]]
            # protocol[i] = protocols[subject_IDs[i]]
            # hand[i] = hands[subject_IDs[i]]

        self.y = y
        self.site = site

        phonetic_data = np.zeros([num_nodes, 5+16+11+17+3], dtype=np.float32)
        phonetic_data[:, 0] = edu
        phonetic_data[:, 1] = gender
        phonetic_data[:, 2] = age
        phonetic_data[:, 3] = site
        phonetic_data[:, 4] = duration

        phonetic_data[:, 5:5+16] = qles
        phonetic_data[:, 5+16:5+16+11] = ymsr
        phonetic_data[:, 5+16+11:5+16+11+17] = hamd


        phonetic_data[:,-3] = np.sum(qles,-1,keepdims=False)
        phonetic_data[:,-2] = np.sum(ymsr,-1,keepdims=False)
        phonetic_data[:,-1] = np.sum(hamd,-1,keepdims=False)

        # phonetic_data[:,3] = protocol
        # phonetic_data[:,4] = hand
        


        self.pd_dict['edu'] = np.copy(phonetic_data[:, 0])
        self.pd_dict['sex'] = np.copy(phonetic_data[:, 1])
        self.pd_dict['age'] = np.copy(phonetic_data[:, 2])
        self.pd_dict['site'] = np.copy(phonetic_data[:,3])
        self.pd_dict['duration'] = np.copy(phonetic_data[:,4])

        if opt.use_qn:
            self.pd_dict['qles_sum'] = np.copy(np.sum(phonetic_data[:,5:5+16],axis=-1))
            self.pd_dict['ymsr_sum'] = np.copy(np.sum(phonetic_data[:,5+9:5+16+11],axis=-1))
            self.pd_dict['hamd_sum'] = np.copy(np.sum(phonetic_data[:,5+9+11:5+16+11+17],axis=-1))

        # self.pd_dict['hand'] = np.copy(phonetic_data[:,4])

        if not opt.use_qn:
            phonetic_data = phonetic_data[:,:5]

        self.pheno = phonetic_data
        # feature_matrix, label: (0 or -1), phonetic_data.shape = (num_nodes, num_phonetic_dim)
        print('pheno data dim:',phonetic_data.shape)
        return subject_IDs,self.y, phonetic_data,site_num,site

    def data_split(self, n_folds,train_val_num):
        # split data by k-fold CV
        n_sub = train_val_num # train HC:MDD=416:186 new signal
        id = list(range(n_sub))
        import random
        # random.seed(321)
        # random.shuffle(id)

        kf = KFold(n_splits=n_folds, random_state=123,shuffle = True) # random state=321
        
        train_index = list()
        val_index = list()

        for tr,va in kf.split(np.array(id)):
            val_index.append(va)
            train_index.append(tr)
            

        train_id = train_index
        val_id = val_index

        return train_id,val_id
    
    def data_split_loso(self):
        # split data by site
        train_index = list()
        val_index = list()
        # loso cv
        site_num = len(np.unique(self.pheno[:,0])) 
        for site in range(site_num):
            test_ind = np.array(np.where(self.pheno[:,0]==site)).squeeze()
            train_ind = [ind for ind in range(self.pheno.shape[0]) if ind not in test_ind]
            train_index.append(train_ind)
            val_index.append(test_ind)

        return [train_index,val_index]

    def data_split_site(self):
        # split data by site
        train_index = list()
        val_index = list()
        train_inds,test_inds = [],[]

        site_test = [2] 

        for site in site_test:
            test_inds.extend(np.array(np.where(self.pheno[:,0]==site)).squeeze())
        train_inds.extend([ind for ind in range(self.pheno.shape[0]) if ind not in test_inds])
        train_index.append(np.array(train_inds))
        val_index.append(np.array(test_inds))

        return [train_index,val_index]


    def get_PAE_inputs(self, nonimg,embeddings):
        '''
        get PAE inputs 
        nonimg: N sub x nonimg vector

        return:
            clinical similarity matrix
        '''
        # construct edge network inputs
        n = embeddings.shape[0]
        node_ftr = np.array(embeddings.detach().cpu().numpy())
        num_edge = n * (1 + n) // 2 - n  
        pd_ftr_dim = nonimg.shape[1] # phenotypic feature dim
        edge_index = np.zeros([2, num_edge], dtype=np.int64)
        edgenet_input = np.zeros([num_edge, 2 * pd_ftr_dim], dtype=np.float32) 
        aff_score = np.zeros(num_edge, dtype=np.float32)
        # static affinity score used to pre-prune edges
        aff_adj = Reader.get_static_affinity_adj(node_ftr, self.pd_dict,opt.use_qn,opt.use_duration) 
        flatten_ind = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_index[:, flatten_ind] = [i, j] 
                edgenet_input[flatten_ind] = np.concatenate((nonimg[i], nonimg[j]))
                aff_score[flatten_ind] = aff_adj[i][j] 
                # print(aff_score[flatten_ind])
                flatten_ind += 1

        assert flatten_ind == num_edge, "Error in computing edge input"

        keep_ind = np.where(aff_score > opt.pheno_edge_threshold)[0] 
        # print('pheno edge kept:', len(keep_ind))
        edge_index = edge_index[:, keep_ind]
        edgenet_input = edgenet_input[keep_ind]

        return edge_index, edgenet_input


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def prepare_local_data(timeseries_path,t1_root):
    data = np.load(timeseries_path, allow_pickle=True).item()
    final_fc = data["timeseires"]
    final_pearson = data["corr"]
    labels = data["label_relief"]
    labels[labels == -1] = 0
    sub_names = data['sub_name']
    

    # select out balanced ID
    if opt.use_all:
        if 'treatment' in timeseries_path: sub_txt_path = "H:/treatment/rest_final/subIDs_all.txt" 
    else:
        if 'treatment' in timeseries_path: sub_txt_path = "./subIDs.txt" 
    

    balanced_names = np.genfromtxt(sub_txt_path, dtype=str) # REST_meta_MDD
    print('subject loaded:', sub_txt_path)

    if  opt.one_per_sub and 'treatment' in timeseries_path:
        balanced_ind = []
        balanced_names = list(balanced_names)
        for index,name in enumerate(sub_names):
            if name in balanced_names:
                balanced_ind.append(index)
                balanced_names.remove(name)
    else:
        balanced_ind = [index  for index,name in enumerate(sub_names) if name in balanced_names]

    # balanced_ind = [index  for index,name in enumerate(sub_names) if name in balanced_names]
    random.seed(123)
    random.shuffle(balanced_ind)
    final_fc = final_fc[balanced_ind]
    labels = labels[balanced_ind]
    sub_names = sub_names[balanced_ind]
    # no combat
    final_pearson = final_pearson[balanced_ind] # no combat
    
    str_names = sub_names
    

    int_names = []
    for name in sub_names:
        if 'YD' in name:
            int_names.append(int(name[-3:])+1000)
        else:
            int_names.append(int(name[-3:]))
    sub_names = np.array(int_names,dtype=np.int32)
    
    final_fc_z = final_fc

    
    

    final_fc = final_fc_z

    final_fc = np.array(final_fc)
    print('nan in final_fc:',np.isnan(final_fc).any())
   
    
    final_fc, final_pearson, labels = [torch.from_numpy(data).float() for data in (final_fc, final_pearson, labels)]
    

    return final_fc, final_pearson, labels,torch.from_numpy(sub_names),str_names
    

def prepare_local_dataloader(timeseries_path,t1_root):
    all_final_fc = []
    all_final_pearson = []
    all_labels = []
    all_idx_names = []
    all_t1_feature = []
    all_str_names = []

    for i in range(len(timeseries_path)):
        final_fc, final_pearson, labels,idx_names,str_names = prepare_local_data(timeseries_path[i],t1_root[i])
        if len(all_idx_names) != 0: idx_names = idx_names + len(all_idx_names[-1])
        all_final_fc.append(final_fc[:,:,:config['data']['window_width']])
        all_final_pearson.append(final_pearson)
        all_labels.append(labels)
        all_idx_names.append(idx_names)
        
        all_str_names.append(str_names)
    
    final_fc = torch.cat(all_final_fc,dim=0)
    final_pearson = torch.cat(all_final_pearson)
    labels = torch.cat(all_labels)
    idx_names = torch.cat(all_idx_names)
    str_names = np.concatenate(all_str_names)
    
    dataset = utils.TensorDataset(
        final_fc,
        final_pearson,
        labels,
        idx_names,
    )
    
    local_dataloader = utils.DataLoader(dataset, batch_size=config["data"]["batch_size"], shuffle=False, drop_last=False)

    return local_dataloader,str_names

def min_max_normalize(data):
    
    min_vals = np.min(data, axis=1,keepdims=True)
    max_vals = np.max(data, axis=1,keepdims=True)
    
    normalized_data = (data - min_vals) / (max_vals - min_vals)
    
    return normalized_data

def percentile_rank_normalization(data):
    ranks = np.argsort(np.argsort(data, axis=0), axis=0)
    
    normalized_data = ranks / (data.shape[0] - 1)
    
    return normalized_data



if __name__ == "__main__":
    site = np.zeros([4], dtype=np.int)
    print(site)
    print(site.shape)
