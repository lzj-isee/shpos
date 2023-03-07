from tensorboard.backend.event_processing import event_accumulator
import argparse
from tqdm import tqdm
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--main_folder', type = str, default = 'results_20/results_WineWhite')
    parser.add_argument('--term', type = str, default = 'test_rmse', 
        choices = ['W2', 'test_error', 'test_nll', 'test_rmse'])
    parser.add_argument('--suffix', type = str, default = 'pdf', choices = ['png', 'eps', 'pdf'])
    parser.add_argument('--dpi', type = int, default = 600)
    opts = parser.parse_args()
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    f =  open('./process_results/gen_line.yaml', 'r')
    plot_settings_common = yaml.load(f.read(), Loader=yaml.FullLoader)
    plot_settings_this = plot_settings_common[opts.main_folder]
    log_folder_list = os.listdir(opts.main_folder)
    for name in log_folder_list:
        if os.path.isfile(os.path.join(opts.main_folder, name)):
            log_folder_list.remove(name)
    log_folder_list.sort()
    datas = {}
    # load data
    for log_folder in tqdm(log_folder_list):
        algo_name = log_folder[0:log_folder.find('_')]
        event_folder = os.path.join(opts.main_folder, log_folder)
        files = os.listdir(event_folder)
        for file in files:
            if os.path.splitext(file)[1] in '.md, .npy': continue
            ea = event_accumulator.EventAccumulator(os.path.join(event_folder, file))
            ea.Reload()
            #print(ea.scalars.Keys()) 
            term = ea.scalars.Items(opts.term)
        value = np.array([i.value for i in term])
        if not algo_name in datas: 
            datas[algo_name] = [value]
        else:
            datas[algo_name].append(value)
    # process data
    stds, means = {}, {}
    x_axis, x_index = None, None
    x_axis_SVRG, x_index_SVRG = None, None
    for name in datas.keys():
        datas[name] = np.array(datas[name])
        stds[name] = np.std(datas[name], axis = 0, ddof = 1)
        means[name] = np.mean(datas[name], axis = 0)
        point_num = int(len(np.mean(datas[name], axis = 0))*plot_settings_this['x_ratio'])
        index = np.linspace(0, point_num-1, int(point_num*plot_settings_this['sparse']),dtype=int)
        means[name] = means[name][0: point_num][index]
        x_axis = np.linspace(0, plot_settings_this['x_max'], len(means[name]))
        if name == 'SVRGPOS':
            x_axis_SVRG = np.linspace(1, plot_settings_this['x_max'], len(means[name]))
            continue
    # start plotting
    plt.figure(figsize=(7.68, 4.8))
    for name in plot_settings_common['order']:
        if name not in datas.keys(): continue
        if name == 'SVRGPOS':
            plt.plot(x_axis_SVRG, means[name], 
            color = plot_settings_common['color'][name],
            linestyle = plot_settings_common['linestyle'][name],
            label = plot_settings_common['label'][name],
            alpha = 0.8,
            linewidth = 1)
            plt.xlim(0, plot_settings_this['x_max'])
            continue
        plt.plot(x_axis, means[name], 
            color = plot_settings_common['color'][name],
            linestyle = plot_settings_common['linestyle'][name],
            label = plot_settings_common['label'][name],
            alpha = 0.8,
            linewidth = 1)
        plt.xlim(0, plot_settings_this['x_max'])
        #plt.fill_between(x_axis,
        #    means[name] - 1.96*stds[name]/np.sqrt(len(datas[name])),
        #    means[name] + 1.96*stds[name]/np.sqrt(len(datas[name])),
        #    color = plot_settings_common['color'][name], 
        #    alpha = 0.2)
    plt.legend(fontsize = 18)
    plt.ylim(plot_settings_this['y_min'][opts.term],plot_settings_this['y_max'][opts.term])
    plt.yscale(plot_settings_this['y_scale'][opts.term])
    plt.xlabel(plot_settings_this['x_label'], {'size': 18})
    plt.ylabel(plot_settings_this['y_label'][opts.term], {'size': 18})
    if plot_settings_this['use_tick'] and plot_settings_this['y_scale'][opts.term] != 'log':
        ax = plt.gca()
        ax.ticklabel_format(style = 'sci', axis = 'y', scilimits = (-2,2))
    #plt.tight_layout()
    plt.tick_params(labelsize = 18)
    if opts.suffix == 'eps':
        ax = plt.gca()
        ax.set_rasterized(True)
    plt.savefig('./figures/{}.{}'.format(
        plot_settings_this['output'][opts.term], opts.suffix), dpi = opts.dpi, bbox_inches = 'tight')
    plt.close()

    