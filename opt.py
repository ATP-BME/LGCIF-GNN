import os
import datetime
import argparse
import random
import numpy as np
import torch
import logging
import logging.config

class OptInit():
    def __init__(self):
        # dataset
        parser = argparse.ArgumentParser(description='PyTorch implementation of EV-GCN')

        # dataset
        parser.add_argument('--config_filename', default='setting/Treatment_fbnetgen.yaml', type=str, help='Configuration filename for training the model.')

        parser.add_argument('--test dataset', default='test balanced new t1 feature', type=str, help='mode test')
        parser.add_argument('--mode', default='mode6', type=str, help='mode of node feature')
        
        parser.add_argument('--one_per_sub', default=False, type=bool, help='use 1 sample of timeseries per subject')
        parser.add_argument('--use_all', default=False, type=self.str2bool, help='use the entire dataset')
        parser.add_argument('--use_glt', default=False, type = self.str2bool,help='use glt T1 features')

        parser.add_argument('--exp_info', default='info', type=str, help='mode of node feature')
        parser.add_argument('--pheno_edge_threshold', type=float, default=1.1, help='mode of test dataset')

        parser.add_argument('--train', default=1, type=int, help='train(default) or evaluate')
        parser.add_argument('--interpre', default=0, type=int, help='train(default) or evaluate')
        parser.add_argument('--interp_grad', default=False, type=self.str2bool, help='run gradient interpretation')

        parser.add_argument('--construct_graph', default=1, type=int, help='train(default) or evaluate')
        parser.add_argument('--use_cpu', action='store_true', help='use cpu?')

        parser.add_argument('--mixup', default=True, type =self.str2bool,help='use graph mixup')
        parser.add_argument('--mixup_rate', default=0.1, type = float,help='the percentage mixup nodes')

        parser.add_argument('--shift_robust', default=False, type=self.str2bool, help='use shift robust loss')
        parser.add_argument('--shift_loss_weight', default=1, type=float, help='shift loss weight default: 1 ')
        parser.add_argument('--g_closs_w', default=4, type = float,help='weight of global class loss 5')
        parser.add_argument('--use_local_loss', default=True, type=self.str2bool, help='use shift robust loss')
        parser.add_argument('--l_closs_w', default=1, type = float,help='weight of local class loss 0.2')
        parser.add_argument('--use_site_loss', default=False, type=self.str2bool, help='use shift robust loss')
        parser.add_argument('--l_sloss_w', default=0.1, type = float,help='weight of local site loss ')
        parser.add_argument('--com_loss_w', default=0.5, type = float,help='weight of modal common loss 5e-5')
        parser.add_argument('--dep_loss_w', default=1e-4, type = float,help='weight of modal dependent loss 1e-12')
        parser.add_argument('--focal_loss', default=False, type=self.str2bool, help='use focal loss')

        parser.add_argument('--use_qn', default=True, type=self.str2bool, help='use questionaire')
        parser.add_argument('--use_duration', default=False, type=self.str2bool, help='use duration')


        parser.add_argument('--lr', default=5e-3, type=float, help='initial learning rate')
        parser.add_argument('--wd', default=1e-2, type=float, help='weight decay')#5e-5
        parser.add_argument('--num_iter', default=200, type=int, help='number of epochs for training')
        parser.add_argument('--edropout', type=float, default=0.3, help='edge dropout rate')
        parser.add_argument('--dropout', default=0.5, type=float, help='ratio of dropout')
        parser.add_argument('--snowball_layer_num', default=8, type=int, help='num of snowball layer')
        parser.add_argument('--num_classes', type=int, default=2, help='number of classes')
        parser.add_argument('--ckpt_path', type=str, default='./save_models/pre_only', help='checkpoint path to save trained models')
        
        parser.add_argument('--n', type=int, default=None, help='knn')
        parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer')
        parser.add_argument('--stepsize', type=int, default=200, help='scheduler step size')
        parser.add_argument('--gamma', type=float, default=0.5, help='scheduler shrinking rate')
        
        # for PairNorm
        # - PairNorm mode, use PN-SI or PN-SCS for GCN and GAT. With more than 5 layers get lots improvement.
        parser.add_argument('--norm_mode', type=str, default='PN', help='Mode for PairNorm, {None, PN, PN-SI, PN-SCS}')
        parser.add_argument('--norm_scale', type=float, default=4.0, help='Row-normalization scale')
        
        args = parser.parse_args()

        args.time = datetime.datetime.now().strftime("%y%m%d")

        if args.use_cpu:
            args.device = torch.device('cpu')
        else:
            args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            # print(" Using GPU in torch")

        
        self.args = args

    def print_args(self):
        # self.args.printer args
        print("==========       CONFIG      =============")
        for arg, content in self.args.__dict__.items():
            print("{}:{}".format(arg, content))
        print("==========     CONFIG END    =============")
        print("\n")
        phase = 'train' if self.args.train==1 else 'eval'
        print('===> Phase is {}.'.format(phase))

    def initialize(self):
        self.set_seed(123)
        #self.logging_init()
        # self.print_args()
        return self.args

    def set_seed(self, seed=123):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def str2bool(self,str):
	        return True if str.lower() == 'true' else False

