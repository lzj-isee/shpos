from tensorboard.backend.event_processing import event_accumulator
import argparse
from tqdm import tqdm
import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pretty_errors
def str2bool(v):
    return v.lower() in ('true')

if __name__ == "__main__":    
    key_word_list = ['0.0e+00', '1.0e+00', '2.0e+00', '5.0e+00', '1.0e+01', '2.0e+01']
    color_dict = {
        '0.0e+00':'dodgerblue', 
        '1.0e+00':'darkcyan', 
        '2.0e+00':'forestgreen', 
        '5.0e+00':'orange', 
        '1.0e+01':'red', 
        '2.0e+01':'darkgray'
    }
    label_dict = {
        '0.0e+00':'Beta = 0', 
        '1.0e+00':'Beta = 1', 
        '2.0e+00':'Beta = 2', 
        '5.0e+00':'Beta = 5', 
        '1.0e+01':'Beta = 10', 
        '2.0e+01':'Beta = 20'
    }
    parser = argparse.ArgumentParser()
    parser.add_argument('--main_folder', type = str, default = 'results_20/results_toydataset_3_d')
    parser.add_argument('--term', type = str, default = 'W2', 
        choices = ['W2', 'test_error', 'test_nll', 'test_rmse'])
    parser.add_argument('--suffix', type = str, default = 'png', choices = ['png', 'eps', 'pdf'])
    parser.add_argument('--dpi', type = int, default = 600)
    opts = parser.parse_args()
    log_folder_list = os.listdir(opts.main_folder)
    for name in log_folder_list:
        if os.path.isfile(os.path.join(opts.main_folder, name)):
            log_folder_list.remove(name)
    log_folder_list.sort()
    datas = {}
    for log_folder in tqdm(log_folder_list):
        key_word = log_folder[log_folder.find('w[')+2:log_folder.find('w[')+9]
        event_folder = os.path.join(opts.main_folder, log_folder)
        files = os.listdir(event_folder)
        for file in files:
            if os.path.splitext(file)[1] in '.md, .npy': continue
            ea = event_accumulator.EventAccumulator(os.path.join(event_folder, file))
            ea.Reload()
            #print(ea.scalars.Keys()) 
            term = ea.scalars.Items(opts.term)
        value = np.array([i.value for i in term])
        if not key_word in datas: 
            datas[key_word] = [value]
        else:
            datas[key_word].append(value)
    stds, means = {}, {}
    x_axis = None
    for name in datas.keys():
        datas[name] = np.array(datas[name])
        stds[name] = np.std(datas[name], axis = 0, ddof = 1)
        means[name] = np.mean(datas[name], axis = 0)
        x_axis = np.linspace(0, 400, len(means[name]))
    plt.figure(figsize=(5.12, 3.84))
    for name in key_word_list:
        if name not in datas.keys(): continue
        plt.plot(x_axis, means[name], 
            color = color_dict[name],
            linestyle = '-',
            label = label_dict[name],
            alpha = 0.8,
            linewidth = 1)
        plt.fill_between(x_axis,
            means[name] - 1.96*stds[name]/np.sqrt(len(datas[name])),
            means[name] + 1.96*stds[name]/np.sqrt(len(datas[name])),
            color = color_dict[name], 
            alpha = 0.2)
    plt.legend()
    plt.ylim(7, 60)
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('2-Wasserstein distance')
    plt.tight_layout()
    if opts.suffix == 'eps':
        ax = plt.gca()
        ax.set_rasterized(True)
    plt.savefig('./figures/TOY3_d.{}'.format(opts.suffix), dpi = opts.dpi)
    plt.close()