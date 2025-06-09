# Copyright (c) 2019 Mwiza Kunda
# Copyright (C) 2017 Sarah Parisot <s.parisot@imperial.ac.uk>, Sofia Ira Ktena <ira.ktena@imperial.ac.uk>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implcd ied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import os
import warnings
import glob
import csv
import numpy as np
import scipy.io as sio
from nilearn import connectome
import pandas as pd
from scipy.spatial import distance
from scipy import signal
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

# Input data variables


class Reader:

    def __init__(self, root_path, id_file_path=None) -> None:

        root_folder = root_path
        
        self.data_folder = root_path
        print('root path:',self.data_folder)

        self.id_file = id_file_path


    def fetch_filenames(self, subject_IDs):
        """
            subject_list : list of short subject IDs in string format
            file_type    : must be one of the available file types
            filemapping  : resulting file name format
        returns:
            filenames    : list of filetypes (same length as subject_list)
        """
        # The list to be filled
        filenames = []

        print('searching in:',self.data_folder)
        file_list = os.listdir(self.data_folder)


        # Fill list with requested file paths
        for i in range(len(subject_IDs)):
            find_file = [file  for file in file_list if file.startswith('ROISignals_ROISignal_{}'.format(subject_IDs[i]))]

            if len(find_file) > 0:
                find_file = os.path.join(self.data_folder,find_file[0])
                filenames.append(find_file)
            else:
                print('Error! file not found:',find_file)
                filenames.append('')

        return filenames


    # Get timeseries arrays for list of subjects
    def get_timeseries(self, subject_list, atlas_name, silence=False):
        """
            subject_list : list of short subject IDs in string format
            atlas_name   : the atlas based on which the timeseries are generated e.g. aal, cc200
        returns:
            time_series  : list of timeseries arrays, each of shape (timepoints x regions)
        """

        timeseries = []
        for i in range(len(subject_list)):
            subject_folder = os.path.join(self.data_folder, subject_list[i])
            ro_file = [f for f in os.listdir(subject_folder) if f.startswith('ROISignal_' + atlas_name)]
            fl = os.path.join(subject_folder, ro_file[0])
            sub_timeseries = sio.loadmat(fl)['ROISignals']
            if silence != True:
                print("Reading timeseries file %s" % fl)
            timeseries.append(sub_timeseries)

        return timeseries


    #  compute connectivity matrices
    def subject_connectivity(self, timeseries, subjects, atlas_name, kind, iter_no='', seed=1234,
                            n_subjects='', save=True, save_path=None):
        """
            timeseries   : timeseries table for subject (timepoints x regions)
            subjects     : subject IDs
            atlas_name   : name of the parcellation atlas used
            kind         : the kind of connectivity to be used, e.g. lasso, partial correlation, correlation
            iter_no      : tangent connectivity iteration number for cross validation evaluation
            save         : save the connectivity matrix to a file
            save_path    : specify path to save the matrix if different from subject folder
        returns:
            connectivity : connectivity matrix (regions x regions)
        """

        if kind in ['TPE', 'TE', 'correlation','partial correlation']:
            if kind not in ['TPE', 'TE']:
                conn_measure = connectome.ConnectivityMeasure(kind=kind)
                connectivity = conn_measure.fit_transform(timeseries)
            else:
                if kind == 'TPE':
                    conn_measure = connectome.ConnectivityMeasure(kind='correlation')
                    conn_mat = conn_measure.fit_transform(timeseries)
                    conn_measure = connectome.ConnectivityMeasure(kind='tangent')
                    connectivity_fit = conn_measure.fit(conn_mat)
                    connectivity = connectivity_fit.transform(conn_mat)
                else:
                    conn_measure = connectome.ConnectivityMeasure(kind='tangent')
                    connectivity_fit = conn_measure.fit(timeseries)
                    connectivity = connectivity_fit.transform(timeseries)

        if save:
            if not save_path:
                save_path = self.data_folder
            if kind not in ['TPE', 'TE']:
                for i, subj_id in enumerate(subjects):
                    subject_file = os.path.join(save_path, subj_id,
                                                subj_id + '_' + atlas_name + '_' + kind.replace(' ', '_') + '.mat')
                    sio.savemat(subject_file, {'connectivity': connectivity[i]})
                return connectivity
            else:
                for i, subj_id in enumerate(subjects):
                    subject_file = os.path.join(save_path, subj_id,
                                                subj_id + '_' + atlas_name + '_' + kind.replace(' ', '_') + '_' + str(
                                                    iter_no) + '_' + str(seed) + '_' + validation_ext + str(
                                                    n_subjects) + '.mat')
                    sio.savemat(subject_file, {'connectivity': connectivity[i]})
                return connectivity_fit


    # Get the list of subject IDs

    def get_ids(self, num_subjects=None):
        """
        return:
            subject_IDs    : list of all subject IDs
        """

        subject_IDs = np.genfromtxt(self.id_file, dtype=str)

        if num_subjects is not None:
            subject_IDs = subject_IDs[:num_subjects]

        return subject_IDs



    def extract_label(self,sub,score):
        scores_dict = {}
        excel_path = r"./clinical_info.xlsx"
        df = pd.read_excel(excel_path)
        for a_sub in sub:
            # 找到IPID列中值为search_value的行
            filtered_df = df[df['IPID'] == a_sub]
            scores_dict[a_sub] = filtered_df[score].item()
        
        return scores_dict

    # preprocess phenotypes. Categorical -> ordinal representation
    @staticmethod
    def preprocess_phenotypes(pheno_ft, params):
        if params['model'] == 'MIDA':
            ct = ColumnTransformer([("ordinal", OrdinalEncoder(), [0, 1, 2])], remainder='passthrough')
        else:
            ct = ColumnTransformer([("ordinal", OrdinalEncoder(), [0, 1, 2, 3])], remainder='passthrough')

        pheno_ft = ct.fit_transform(pheno_ft)
        pheno_ft = pheno_ft.astype('float32')

        return (pheno_ft)


    # create phenotype feature vector to concatenate with fmri feature vectors
    @staticmethod
    def phenotype_ft_vector(pheno_ft, num_subjects, params):
        gender = pheno_ft[:, 0]
        if params['model'] == 'MIDA':
            eye = pheno_ft[:, 0]
            hand = pheno_ft[:, 2]
            age = pheno_ft[:, 3]
            fiq = pheno_ft[:, 4]
        else:
            eye = pheno_ft[:, 2]
            hand = pheno_ft[:, 3]
            age = pheno_ft[:, 4]
            fiq = pheno_ft[:, 5]

        phenotype_ft = np.zeros((num_subjects, 4))
        phenotype_ft_eye = np.zeros((num_subjects, 2))
        phenotype_ft_hand = np.zeros((num_subjects, 3))

        for i in range(num_subjects):
            phenotype_ft[i, int(gender[i])] = 1
            phenotype_ft[i, -2] = age[i]
            phenotype_ft[i, -1] = fiq[i]
            phenotype_ft_eye[i, int(eye[i])] = 1
            phenotype_ft_hand[i, int(hand[i])] = 1

        if params['model'] == 'MIDA':
            phenotype_ft = np.concatenate([phenotype_ft, phenotype_ft_hand], axis=1)
        else:
            phenotype_ft = np.concatenate([phenotype_ft, phenotype_ft_hand, phenotype_ft_eye], axis=1)

        return phenotype_ft


    # Load precomputed fMRI connectivity networks
    def get_networks(self, subject_list, kind, iter_no='', seed=1234, n_subjects='', atlas_name="aal",
                    variable='connectivity'):
        """
            subject_list : list of subject IDs
            kind         : the kind of connectivity to be used, e.g. lasso, partial correlation, correlation
            atlas_name   : name of the parcellation atlas used
            variable     : variable name in the .mat file that has been used to save the precomputed networks
        return:
            matrix      : feature matrix of connectivity networks (num_subjects x network_size)
        """

        all_networks = []
        for subject in subject_list:
            if len(kind.split()) == 2:
                kind = '_'.join(kind.split())
            fl = os.path.join(self.data_folder, subject,
                                subject + "_" + atlas_name + "_" + kind.replace(' ', '_') + ".mat")


            matrix = sio.loadmat(fl)[variable]
            all_networks.append(matrix)

        if kind in ['TE', 'TPE']:
            norm_networks = [mat for mat in all_networks]
        else:
            norm_networks = [np.arctanh(mat) for mat in all_networks]

        networks = np.stack(norm_networks)

        return networks

    def pad_to_time_size(self,x, time_size):
        assert x.ndim == 2
        if x.shape[0] >= time_size:
            return x[:time_size,:]
        else:
            return np.pad(x, ((0,time_size-x.shape[0]%time_size),(0,0)), 'wrap')

    def pad_to_length(self,x, length):
        assert x.ndim == 3
        assert x.shape[-1] <= length
        if x.shape[-1] == length:
            return x

        return np.pad(x, ((0,0),(0,0), (0, length - x.shape[-1])), 'wrap')

    def normalize(self,x, mean=None, std=None):
        '''
        x: [n time points, n voxels]
        '''
        # nan_positions = np.argwhere(np.isnan(x))
        # print(nan_positions)
        if np.isnan(x).any():
            # 计算每个voxel的平均值
            voxel_means = np.nanmean(x, axis=1)

            # 用平均值替换NaN
            for i in range(x.shape[0]):
                x[i, np.isnan(x[i, :])] = voxel_means[i]

        mean = np.mean(x) if mean is None else mean
        std = np.std(x) if std is None else std
        return (x - mean) / (std * 1.0)

    def normalize_channel(self, x):
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

