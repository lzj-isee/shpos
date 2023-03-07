import pretty_errors
import argparse
import importlib
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # basic setting
    parser.add_argument('--algorithm', type = str, default = 'SPOSHMC2_VR') 
    parser.add_argument('--task', type = str, default = 'fnn2_regression_g')
    parser.add_argument('--dataset', type = str, default='abalone')
    parser.add_argument('--epochs', type = int, default = 10)
    parser.add_argument('--pre_batch_size', type = int, default = 1000, help = 'init batchsize for some VR algo')
    parser.add_argument('--eval_interval', type = int, default = 1)
    parser.add_argument('--save_folder', type=str, default='results')
    parser.add_argument('--gpu_ids', type=str, default='3',help='gpu ids: e.g. 0,1,2. -1 for cpu')
    parser.add_argument('--seed', type=int, default = 0)
    parser.add_argument('--split_seed', type = int, default = 9)
    opts,_ = parser.parse_known_args()
    # set gpu ids
    str_ids = opts.gpu_ids.split(',')
    opts.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            opts.gpu_ids.append(id)
    if len(opts.gpu_ids) > 0:
        torch.cuda.set_device(opts.gpu_ids[0])
    # import algorithm settings
    algorithm_settings = importlib.import_module('settings.algorithms.{:}'.format(opts.algorithm))
    parser = algorithm_settings.__getattribute__('args')(parser).return_parser()
    opts,_ = parser.parse_known_args()
    # import task settings
    task_settints = importlib.import_module('settings.task.{:}'.format(opts.task))
    parser = task_settints.__getattribute__('args')(parser).return_parser()
    opts = parser.parse_args()
    # import algorithm
    algorithm = importlib.import_module('algorithms.{:}'.format(opts.algorithm))
    algorithm.__getattribute__('{:}'.format(opts.algorithm))(opts).start_sample()
