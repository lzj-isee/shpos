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
import matplotlib as mpl
def str2bool(v):
    return v.lower() in ('true')

if __name__ == "__main__":  
    # settings
    var = 6
    cov = 0.9 * 6
    mean = torch.zeros(1,2)
    index_list = [0, 1, 2, 5, 10, 20, 40]
    key_word_list = ['SPOS', 'SPOSHMC2']
    label_dict = {
        'SPOS': 'SPOS',
        'SPOSHMC2': 'H-SPOS',
    }
    parser = argparse.ArgumentParser()
    parser.add_argument('--main_folder', type = str, default = 'results_20/results_toydataset_3')
    parser.add_argument('--suffix', type = str, default = 'eps', choices = ['png', 'eps', 'pdf'])
    parser.add_argument('--dpi', type = int, default = 600)
    opts = parser.parse_args()
    #mpl.rcParams['text.usetex'] = True
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
    def pdf_calc_2dim(x):  # x: M * dim matrix
        result = torch.exp(
            - (torch.matmul((x-mean), in_cov_matrix) * (x-mean)).sum(1) / 2)
        return result   # M array
    x, y = torch.linspace(-6, 6, 121), torch.linspace(-6, 6, 121)
    grid_X, grid_Y = torch.meshgrid(x,y)
    loc = torch.cat([grid_X.unsqueeze(2), grid_Y.unsqueeze(2)], dim = 2)
    num = len(loc)
    pdf = pdf_calc_2dim(loc.view(-1, 2))
    pdf = pdf.view(num, num)
    for name in tqdm(key_word_list):
        for index in index_list:
            samples = datas[name][index]
            plt.figure(figsize=(5.12, 3.84))
            plt.contour(grid_X.cpu().numpy(), grid_Y.cpu().numpy(), pdf.cpu().numpy(),
                cmap = 'hot', linewidths = 1, alpha = 0.8)
            plt.scatter(samples[:, 0], samples[:, 1],alpha = 0.5, s = 15, label = label_dict[name])
            plt.legend()
            plt.ylim([ -6, 6])
            plt.xlim([ -6, 6])
            plt.tight_layout()
            if opts.suffix == 'eps':
                ax = plt.gca()
                ax.set_rasterized(True)
            plt.savefig('./figures/toy3/{}.{}'.format(name+'_'+str(index*10+1), opts.suffix), dpi = opts.dpi)
            plt.close()