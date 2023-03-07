from tensorboard.backend.event_processing import event_accumulator
import argparse
from tqdm import tqdm
import torch
import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pretty_errors
import matplotlib
def str2bool(v):
    return v.lower() in ('true')

if __name__ == "__main__":  
    np.random.seed(0)
    # settings
    var = 6
    cov = -0.98 * 6
    mu = 2  
    index_list = [0, 1, 2, 5, 10, 20, 40]
    key_word_list = ['SVGD', 'SPOS', 'HMCSPOS', 'WNes', 'ParaHMC']
    label_dict = {
        'SVGD': 'SVGD',
        'SPOS': 'SPOS',
        'HMCSPOS': 'SHPOS',
        'WNes': 'SVGD-WNes',
        'ParaHMC': 'UL-MCMC'
    }
    parser = argparse.ArgumentParser()
    parser.add_argument('--main_folder', type = str, default = 'results_20/results_toy1')
    parser.add_argument('--suffix', type = str, default = 'pdf', choices = ['png', 'eps', 'pdf'])
    parser.add_argument('--dpi', type = int, default = 600)
    opts = parser.parse_args()
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    # filt files
    log_folder_list = []
    for name in os.listdir(opts.main_folder):
        if 's[0]' in name:
            log_folder_list.append(name)
    log_folder_list.sort()
    datas = {}
    # load data
    for log_folder in tqdm(log_folder_list):
        algo_name = log_folder[0:log_folder.find('_')]
        event_folder = os.path.join(opts.main_folder, log_folder)
        files = os.listdir(event_folder)
        for file in files:
            if os.path.splitext(file)[1] == '.npy': 
                datas[algo_name] = np.load(os.path.join(opts.main_folder, log_folder, file))
    # pre calc before plot
    cov_matrix = torch.Tensor([[var, cov],[cov, var]])
    in_cov_matrix = torch.inverse(cov_matrix)
    @torch.no_grad()
    def pdf_calc(x):  # x: M * dim matrix
        result = torch.exp( - (torch.matmul(x, in_cov_matrix)* x ).sum(1) / 2) + \
            0.5 * torch.exp( - (torch.matmul((x-mu), in_cov_matrix)*(x-mu)).sum(1) / 2) + \
            0.5 * torch.exp( - (torch.matmul((x+mu), in_cov_matrix)*(x+mu)).sum(1) / 2)
        return result   # M array
    x, y = torch.linspace(-6, 6, 121), torch.linspace(-6, 6, 121)
    grid_X, grid_Y = torch.meshgrid(x,y)
    loc = torch.cat([grid_X.unsqueeze(2), grid_Y.unsqueeze(2)], dim = 2)
    num = len(loc)
    pdf = pdf_calc(loc.view(-1, 2))
    pdf = pdf.view(num, num)
    init_samples = np.random.standard_normal((1000,2)) * 0.25 + np.array([[-4 , 2]])
    for name in tqdm(key_word_list):
        # plot the init setting
        plt.figure(figsize=(7.2, 4.8))
        plt.contour(grid_X.cpu().numpy(), grid_Y.cpu().numpy(), pdf.cpu().numpy(),
                cmap = 'hot', linewidths = 1, alpha = 0.8)
        plt.scatter(init_samples[:, 0], init_samples[:, 1],alpha = 0.5, s = 15, label = label_dict[name]+', iteration = 0')
        plt.legend(fontsize = 22, loc = 3)
        plt.ylim([ -6, 6])
        plt.xlim([ -6, 6])
        plt.tick_params(labelsize = 22)
        plt.tight_layout()
        if opts.suffix == 'eps':
            ax = plt.gca()
            ax.set_rasterized(True)
        plt.savefig('./figures/toy1/{}.{}'.format(name+'_0', opts.suffix), dpi = opts.dpi, bbox_inches = 'tight')
        plt.close()
        # plot the iteration
        for index in index_list:
            samples = datas[name][index]
            plt.figure(figsize=(7.2, 4.8))
            plt.contour(grid_X.cpu().numpy(), grid_Y.cpu().numpy(), pdf.cpu().numpy(),
                cmap = 'hot', linewidths = 1, alpha = 0.8)
            plt.scatter(samples[:, 0], samples[:, 1],alpha = 0.5, s = 15, label = label_dict[name]+', iteration = {}'.format(index*10+1))
            plt.legend(fontsize = 22, loc = 3)
            plt.ylim([ -6, 6])
            plt.xlim([ -6, 6])
            plt.tick_params(labelsize = 22)
            plt.tight_layout()
            if opts.suffix == 'eps':
                ax = plt.gca()
                ax.set_rasterized(True)
            plt.savefig('./figures/toy1/{}.{}'.format(name+'_'+str(index*10+1), opts.suffix), dpi = opts.dpi, bbox_inches = 'tight')
            plt.close()
    