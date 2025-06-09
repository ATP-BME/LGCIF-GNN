import os
import csv
import numpy as np
import scipy.io as sio
import torch.nn.functional as F
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import RFE
from nilearn import connectome
from scipy.spatial import distance
import SimpleITK as sitk
import pandas as pd

# Reading and computing the input data

# Selected pipeline
pipeline = 'cpac'

# Get the list of subject IDs
def get_ids(num_subjects=None):
    """

    return:
        subject_IDs    : list of all subject IDs
    """

    subject_IDs = np.genfromtxt(os.path.join("G:/treatment/rest_final/subIDs.txt"), dtype=str)

    if num_subjects is not None:
        subject_IDs = subject_IDs[:num_subjects]

    return subject_IDs



def extract_label(sub,score):
        scores_dict = {}
        excel_path = r"H:\treatment\rest_final\164例基线临床数据.xlsx"
        df = pd.read_excel(excel_path)
        for a_sub in sub:
            if score in ['age','sex','edu','site','缓解','有效','Duration_day']:
                # 找到IPID列中值为search_value的行
                filtered_df = df[df['IPID'] == a_sub]
                if filtered_df[score].isna().all():
                    scores_dict[a_sub] = df[score].mean(skipna=True)
                else:
                    scores_dict[a_sub] = filtered_df[score].item()
            else:
                filtered_df = df[df['IPID'] == a_sub]
                if score == 'QLES':
                    scores_dict[a_sub] = []
                    for score_item in range(1,17):
                        if filtered_df['QLES{}'.format(score_item)].isna().all():
                            scores_dict[a_sub].append(df['QLES{}'.format(score_item)].mean(skipna=True))
                        else:
                            scores_dict[a_sub].append(filtered_df['QLES{}'.format(score_item)].item())
                        if scores_dict[a_sub][-1] == ' ':
                            scores_dict[a_sub][-1] = 0
                if score == 'YMSR':
                    scores_dict[a_sub] = []
                    for score_item in range(1,12):
                        if filtered_df['YMSR{}'.format(score_item)].isna().all():
                            scores_dict[a_sub].append(df['YMSR{}'.format(score_item)].mean(skipna=True))
                        else:
                            scores_dict[a_sub].append(filtered_df['YMSR{}'.format(score_item)].item())
                        if scores_dict[a_sub][-1] == ' ':
                            scores_dict[a_sub][-1] = 0
                if score == 'HAMD':
                    scores_dict[a_sub] = []
                    for score_item in range(1,18):
                        if filtered_df['HAMD{}'.format(score_item)].isna().all():
                            scores_dict[a_sub].append(df['HAMD{}'.format(score_item)].mean(skipna=True))
                        else:
                            scores_dict[a_sub].append(filtered_df['HAMD{}'.format(score_item)].item())
                        if scores_dict[a_sub][-1] == ' ':
                            scores_dict[a_sub][-1] = 0
        return scores_dict


def create_affinity_graph_from_scores(scores, pd_dict):
    '''
    phenotypic feature # site sex age protocol hand
    '''
    num_nodes = len(pd_dict[scores[0]])
    graph = np.zeros((num_nodes, num_nodes)) # Adjacent matrix

    for l in scores: # l：表型特征指标
        label_dict = pd_dict[l]

        # if l in ['Age','Education (years)']:
        if l in ['age','edu']:
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    try:
                        val = abs(float(label_dict[k]) - float(label_dict[j])) 
                        if val < 2: 
                            graph[k, j] += 1
                            graph[j, k] += 1
                    except ValueError:  # missing label
                        pass
        if l in ['duration']:
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    try:
                        val = abs(float(label_dict[k])/30 - float(label_dict[j])/30) 
                        if val < 6: 
                            graph[k, j] += 1
                            graph[j, k] += 1
                    except ValueError:  # missing label
                        pass
        if l in ['qles_sum','ymsr_sum','hamd_sum']:
            # thre = 10 # default
            if l == 'qles_sum':
                thre = 5
            if l == 'ymsr_sum':
                thre = 4
            if l == 'hamd_sum':
                thre == 8 # 3 
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    try:
                        val = abs(float(label_dict[k]) - float(label_dict[j]))
                        if val < thre:
                            graph[k, j] += 1
                            graph[j, k] += 1
                    except ValueError:  # missing label
                        pass
        else: 
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    if label_dict[k] == label_dict[j]:
                        graph[k, j] += 1
                        graph[j, k] += 1

    return graph

def get_static_affinity_adj(features, pd_dict,use_qn=True,use_duration=True):
    '''
    input:
        features: N x img feature dim, extracted image feature
        pd_dict: phenotypic feature dict
    '''
    
    if use_qn:
        if use_duration:
            pd_affinity = create_affinity_graph_from_scores(['edu','age','sex','qles_sum','ymsr_sum','hamd_sum','duration'], pd_dict) 
            pd_affinity = create_affinity_graph_from_scores(['edu','age','sex','qles_sum','ymsr_sum','hamd_sum'], pd_dict) 

    else:
        if use_duration:
            pd_affinity = create_affinity_graph_from_scores(['edu','age','sex','duration'], pd_dict) 
        else:
            pd_affinity = create_affinity_graph_from_scores(['edu','age','sex'], pd_dict) # 使用站点ID以及量表评分进行预剪枝

    distv = distance.pdist(features, metric='correlation') 
    dist = distance.squareform(distv) 
    sigma = np.mean(dist)
    feature_sim = np.exp(- dist ** 2 / (2 * sigma ** 2)) 
    
    adj = pd_affinity * feature_sim
    
    adj = (adj - adj.min()) / (adj.max()-adj.min())

    return adj 
