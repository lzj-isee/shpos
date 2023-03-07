import torch
import numpy as np
from tqdm import tqdm
from algorithms.basic import basic
from functions import save_final_results

class ParaHMC(basic):
    def __init__(self, opts):
        algo_name = 'ParaHMC'
        save_name = algo_name + '_'+\
            'lr[{:.2e}]M[{}]b[{}]ge[{:.3f}]u[{:.2f}]s[{}]'.format(\
            opts.lr, opts.particle_num, opts.batch_size, opts.ge, opts.u, opts.seed)
        self.gamma = opts.ge / opts.lr
        super(ParaHMC, self).__init__(opts, algo_name, save_name)
        print('\nalgorithm and setting: \n',self.save_name)

    def gen_noise(self, eta, gamma, u, num, dim, device):
        # generate noise used in some HMC algorithms
        g_e = gamma * eta
        var1 = u * (1 - np.exp(-2 * g_e))
        var2 = (u * gamma**(-2)) * \
            (2*g_e + 4*np.exp(-g_e) - np.exp(-2*g_e) - 3)
        corr = (u * gamma**(-1)) * \
            (1 - 2*np.exp(-g_e) + np.exp(-2*g_e))
        xi = torch.randn(size = (num, dim), device = device)
        xi_v = xi * np.sqrt(var1)
        xi_x = corr/var1 * xi_v + np.sqrt(var2 - corr**2/var1) * torch.randn(size = (num, dim), device=device)
        return xi_x, xi_v

    def start_sample(self):
        x_list = self.functions.init_net((self.opts.particle_num, self.model_dim))
        v_list = torch.zeros_like(x_list)
        grad_list = torch.zeros_like(x_list)
        curr_iter_count = 0
        results = None
        # start updating
        for epoch in tqdm(range(self.opts.epochs)):
            for iter, (train_features, train_labels) in enumerate(self.train_loader):
                curr_iter_count += 1
                # calc grad
                grad_list = self.functions.nl_grads_calc(
                    x = x_list, 
                    features = train_features, 
                    labels = train_labels
                )
                eta = self.opts.lr
                # gen noise
                noise_x, noise_v = self.gen_noise(
                    eta = self.opts.lr,
                    gamma = self.gamma,
                    u = self.opts.u,
                    num = self.opts.particle_num,
                    dim = self.model_dim,
                    device = self.device)
                # update x_list
                x_list += v_list * eta + noise_x
                # update v_list
                v_list += - self.gamma * eta * v_list - eta * self.opts.u * grad_list + noise_v
                # evaluation
                if (curr_iter_count - 1) % self.opts.eval_interval == 0:
                    results = self.functions.evaluation_particles(x_list)
                    self.functions.save_eval_to_tensorboard(
                        self.writer, 
                        results, 
                        epoch*len(self.train_loader)+iter
                    )
                if x_list.max() != x_list.max(): raise ValueError('Nan')  # ensurance
        self.functions.save_final_results(self.writer, self.save_folder, results)
        self.post_process()