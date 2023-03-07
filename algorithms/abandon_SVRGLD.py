import torch
import numpy as np
import os
from tqdm import tqdm
from algorithms.basic import basic

class SVRGLD(basic):
    def __init__(self, opts):
        super(SVRGLD,self).__init__(opts)
        self.opts = opts
        self.algo_name = 'SVRGLD'
        # make sure that batch_size <= data_num
        if opts.batch_size > self.data_num:
            self.batch_size = self.data_num
        else:
            self.batch_size = opts.batch_size
        # set the save_name
        self.save_name=self.algo_name + '_' + \
        'lr[{:.2e}]b[{}]s[{}]'.format(\
            opts.lr, self.batch_size, opts.seed)
        # set save path for figure and sample
        self.save_path_figure = os.path.join(self.save_folder_figures, self.save_name)
        self.save_path_sample = os.path.join(self.save_folder_samples, self.save_name)
        # you can use non-uniform sample if you want
        self.sample_weight = torch.ones(self.data_num, device = self.device)/self.data_num
        print('\nalgorithm and setting: \n',self.save_name)

    # $ grad = grad_{curr} - grad_{snap} + grad_{alpha} $
    def start_sample(self):
        sample_list = torch.zeros((1, self.data_dim), device = self.device)
        curr_point = torch.zeros((1, self.data_dim), device = self.device)
        snap_point = torch.zeros((1, self.data_dim), device = self.device)
        grad_alpha = torch.zeros((1, self.data_dim), device = self.device)
        curr_iter_count = 0
        refresh_count = 0
        # init
        grad_alpha = self.functions.nl_grads_calc(
            loc_point = curr_point,
            data_points = self.data_points
        ).mean(0)
        snap_point = curr_point.clone().detach()
        # start sample
        for _ in tqdm(range(self.opts.iters)):
            # update per epoch
            if (refresh_count * self.batch_size) >= self.data_num:
                refresh_count = 0
                grad_alpha = self.functions.nl_grads_calc(
                    loc_point = curr_point,
                    data_points = self.data_points
                ).mean(0, keepdim = True)
                snap_point = curr_point.clone().detach()
            # update count
            refresh_count += 1
            curr_iter_count += 1
            # sample mini-batch
            s_data_points = self.functions.sample_data(
                weight = self.sample_weight,
                size = self.batch_size
            )
            # calc grad
            grad_curr = self.functions.nl_grads_calc(
                loc_point = curr_point,
                data_points = s_data_points,
            ).mean(0, keepdim = True)
            grad_snap = self.functions.nl_grads_calc(
                loc_point = snap_point,
                data_points = s_data_points
            ).mean(0, keepdim = True)
            grad = grad_curr - grad_snap + grad_alpha
            eta = self.opts.lr
            # gen noise
            noise = torch.randn_like(curr_point, device = self.device) * np.sqrt(2 * eta)
            # update sample point
            curr_point = curr_point - grad * eta + noise
            sample_list = torch.cat([sample_list, curr_point.view(1,-1)], dim = 0)
        sample_array = sample_list.cpu().numpy()
        self.save_results(sample_array, self.save_path_figure, self.save_path_sample)