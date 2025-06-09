import deepdish as dd
import os.path as osp
import os
import numpy as np
import argparse
from pathlib import Path
import pandas as pd
from scipy.io import loadmat

def pad_to_time_size(x, time_size):
    assert x.ndim == 2
    if x.shape[0] >= time_size:
        return x[:time_size,:]
    else:
        return np.pad(x, ((0,time_size-x.shape[0]%time_size),(0,0)), 'wrap')

def pad_to_length(x, length):
    assert x.ndim == 3
    assert x.shape[-1] <= length
    if x.shape[-1] == length:
        return x

    return np.pad(x, ((0,0),(0,0), (0, length - x.shape[-1])), 'wrap')

def normalize(x, mean=None, std=None):
    # nan_positions = np.argwhere(np.isnan(x))
    # print(nan_positions)
    if np.isnan(x).any():
        voxel_means = np.nanmean(x, axis=1)

        for i in range(x.shape[0]):
            x[i, np.isnan(x[i, :])] = voxel_means[i]

    mean = np.mean(x) if mean is None else mean
    std = np.std(x) if std is None else std
    return (x - mean) / (std * 1.0)

def normalize_channel(x):
    '''
    x: [n time points, n voxels]
    '''
    if np.isnan(x).any():
        time_means = np.nanmean(x, axis=0)


        for j in range(x.shape[1]):
            x[np.isnan(x[:, j]), j] = time_means[j]

    mean = np.mean(x, axis=0)  
    std = np.std(x, axis=0)  
    
    std[std == 0] = 1

    normalized_x = (x - mean) / std

    return normalized_x



def main(args):
    
    
    data_dir =  os.path.join("./processed/ROISignal/val_baseline/raw")
    timeseires = os.path.join('./processed/ROISignal/val_baseline')


    times = []

    labels_relief = []
    labels_effective = []

    pcorrs = []

    corrs = []

    site_list = []
    sub_names = []

    subject_IDs = np.genfromtxt("./val_IDs.txt", dtype=str)


    for f in subject_IDs:
        if osp.isfile(osp.join(data_dir, '{}.h5'.format(f))):
            fname = f
            

            files = os.listdir(osp.join(timeseires, fname))

            file = list(filter(lambda x: x.startswith("ROISig"), files))
            file = [atlas_file for atlas_file in file if args.atlas in atlas_file][0]

            time = loadmat(osp.join(timeseires, fname, file))['ROISignals'] # length=195
            time = normalize_channel(time)
            time = pad_to_time_size(time, 200)
            time = time.T
            # pad 195 to 200
            # time = np.pad(time, ((0,0),(0, 5)), mode='symmetric')

            if time.shape[1] < 200:
                print(f,'length not enough')
                continue

            temp = dd.io.load(osp.join(data_dir,  '{}.h5'.format(f)))
            pcorr = temp['pcorr'][()]

            pcorr[pcorr == float('inf')] = 0

            att = temp['corr'][()]

            att[att == float('inf')] = 0

            label_relief = temp['label_relief']
            label_effective = temp['label_effective']


            start = 0
            ww = 200
            window_step = 200
            while start+ww <= time.shape[1]:
                times.append(time[:,start:start+ww])
                labels_relief.append(label_relief[0])
                labels_effective.append(label_effective[0])

                corrs.append(att)
                pcorrs.append(pcorr)
                # site_list.append(int('YD' in f)) # train
                site_list.append(2) # train

                sub_names.append(f)
                start =start + window_step

    np.save('./processed/Anding_post_{}_val.npy'.format(args.atlas), {'timeseires': np.array(times), "label_relief": np.array(labels_relief),"label_effective": np.array(labels_effective),"corr": np.array(corrs),"pcorr": np.array(pcorrs), 'site': np.array(site_list),'sub_name':np.array(sub_names)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate the final dataset')
    parser.add_argument('--root_path', default="", type=str, help='The path of the folder containing the dataset folder.')
    parser.add_argument('--atlas', default="reward70", type=str, help='atlas name')


    args = parser.parse_args()
    main(args)
