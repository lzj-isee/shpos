import os
import torch
import numpy as np
from abc import ABCMeta, abstractclassmethod
import importlib
from functions import create_dirs_if_not_exist, clear_log, save_settings
from libsvm.python.svmutil import *
from tensorboardX import SummaryWriter


class basic(metaclass=ABCMeta):
    def __init__(self, opts, algo_name, save_name):
        self.opts = opts
        self.algo_name = algo_name
        self.save_name = save_name
        # set random seed 
        np.random.seed(opts.seed)
        torch.manual_seed(opts.seed)
        torch.random.manual_seed(opts.seed)
        if torch.cuda.is_available(): torch.cuda.random.manual_seed_all(opts.seed)
        # set device
        self.device = torch.device('cuda:{:}'.format(opts.gpu_ids[0])) \
            if opts.gpu_ids != '-1' else torch.device('cpu')
        # init
        self.save_folder = os.path.join(self.opts.save_folder, self.save_name)
        # import functions by task
        self.functions = importlib.import_module('functions.{:}'.format(self.opts.task))\
            .__getattribute__('functions')(self.opts, self.device)
        if opts.task in 'logistic_regression, fnn2_regression, fnn2_regression_g, toydataset_1, toydataset_3':
            self._configuration_1()


    def _configuration_1(self):
        # load dataset
        if self.opts.task in 'logistic_regression, fnn2_regression, fnn2_regression_g':
            self.functions.load_dataset(self.opts.dataset)
        # set train_loader
        self.functions.set_trainloader()
        # creat log folder
        create_dirs_if_not_exist(self.save_folder)
        # clear old log files
        clear_log(self.save_folder)
        # save settings
        save_settings(self.save_folder, vars(self.opts))
        # creat tensorboard
        self.writer = SummaryWriter(log_dir = self.save_folder)
        # some variables
        if self.opts.task in 'logistic_regression, fnn2_regression, fnn2_regression_g':
            # some variables
            self.train_features, self.train_labels = self.functions.train_features, self.functions.train_labels
            self.test_features, self.test_labels = self.functions.test_features, self.functions.test_labels
            self.train_num, self.test_num = self.functions.train_num, self.functions.test_num
            self.data_dim, self.model_dim = self.functions.data_dim, self.functions.model_dim
            self.train_loader = self.functions.train_loader
            self.train_loader_full = self.functions.train_loader_full
            self.train_iter = iter(self.train_loader)
            self.inner_num = int(self.train_num / self.opts.batch_size)
        elif self.opts.task in 'toydataset_1, toydataset_3':
            self.inner_num = 1
            self.model_dim = self.functions.model_dim
            self.train_loader = self.functions.train_loader
            self.train_iter = iter(self.train_loader)

    def post_process(self):
        self.writer.close()
    
    @abstractclassmethod
    def start_sample(self):
        pass

    