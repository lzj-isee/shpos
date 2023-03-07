import torch
import numpy as np
from tqdm import tqdm
from algorithms.basic import basic

class SGD(basic):
    def __init__(self, opts):
        algo_name = 'SGD'
        save_name = algo_name +'_'+\
        'lr[{:.2e}]b[{}]s[{}]'.format(\
            opts.lr, opts.batch_size, opts.seed)
        super(SGD,self).__init__(opts, algo_name, save_name)
        print('\nalgorithm and setting: \n',self.save_name)

    # $ x_{k+1} = x_k - \eta\nabla{f(x_k)} + \sqrt{2\eta}\epsilon_k $
    def start_sample(self):
        curr_point = self.functions.init_net((self.model_dim, ))
        curr_iter_count = 0
        for _ in tqdm(range(self.opts.iters)):
            curr_iter_count += 1
            # get train data
            try: train_features, train_labels = next(self.train_iter)
            except:
                self.train_iter = iter(self.train_loader)
                train_features, train_labels = next(self.train_iter)
            # calc grad
            grad = self.functions.nl_grads_calc(
                x = curr_point, 
                features = train_features, 
                labels = train_labels
            )
            # update sample point
            curr_point = curr_point - grad * self.opts.lr
            # evaluation
            if (curr_iter_count - 1) % self.opts.eval_interval == 0:
                results = self.functions.evaluation_particles(curr_point)
                self.functions.save_eval_to_tensorboard(self.writer, results, curr_iter_count)
        self.post_process()
        