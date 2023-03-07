from tensorboard.backend.event_processing import event_accumulator
import argparse
import os
import numpy as np
import seaborn
import matplotlib.pyplot as plt
import pandas as pd
def str2bool(v):
    return v.lower() in ('true')


if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type = str, default = 'results_toydataset_3')
    parser.add_argument('--term', type = str, default = 'W2')
    parser.add_argument('--y_min', type = float, default = 7)
    parser.add_argument('--y_max', type = float, default = 80)
    parser.add_argument('--log_y', type = str2bool, default = True)
    opts = parser.parse_args()
    log_folder_list = os.listdir(opts.folder)
    datas = {}
    for log_folder in log_folder_list:
        algo_name = log_folder[0:log_folder.find('_')]
        event_folder = os.path.join(opts.folder, log_folder)
        files = os.listdir(event_folder)
        for file in files:
            if os.path.splitext(file)[1] in '.md, .npy': continue
            ea = event_accumulator.EventAccumulator(os.path.join(event_folder, file))
            ea.Reload()
            print(ea.scalars.Keys()) 
            term = ea.scalars.Items(opts.term)
        value = np.array([i.value for i in term])
        step = np.array([i.step for i in term])
        if not algo_name in datas: 
            datas[algo_name] = [value]
        else:
            datas[algo_name].append(value)
    df = []
    for i, algo in zip(range(len(datas.keys())), datas.keys()):
        values = np.array(datas[algo])
        df.append(pd.DataFrame(values).melt(var_name='Iterations', value_name = opts.term))
        df[i]['Algorithm'] = algo
    df = pd.concat(df)
    seaborn.lineplot(
        x =  'Iterations',
        y = opts.term,
        hue = 'Algorithm',
        style = 'Algorithm',
        data = df
    )
    plt.ylim([opts.y_min, opts.y_max])
    if opts.log_y == True:
        plt.yscale("log")
    plt.savefig('./{}.png'.format(opts.folder), dpi = 500)
    plt.close()