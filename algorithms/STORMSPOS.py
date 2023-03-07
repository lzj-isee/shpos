import torch
import numpy as np
from tqdm import tqdm
from algorithms.basic import basic
from functions import save_final_results

class STORMSPOS(basic):
    def __init__(self, opts):
        algo_name = 'STORMSPOS'
        save_name = algo_name + '_'+\
            'lr[{:.2e}]M[{}]b[{}]k_h[{:.1e}]w[{:.1e}]ge[{:.3f}]u[{:.2f}]rp[{},{:.1f}]s[{}]'.format(\
            opts.lr, opts.particle_num, opts.batch_size, 
            opts.kernel_h, opts.weight, opts.ge, opts.u, 
            opts.refresh_value, opts.pow, opts.seed)
        self.gamma = opts.ge / opts.lr
        super(STORMSPOS, self).__init__(opts, algo_name, save_name)
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
        x_list_last = torch.zeros_like(x_list)
        v_list = torch.zeros_like(x_list)
        grad_list = torch.zeros_like(x_list)
        curr_iter_count = 0
        refresh_count = 1
        results = None
        # start updating
        for epoch in tqdm(range(self.opts.epochs)):
            for iter, (train_features, train_labels) in enumerate(self.train_loader):
                curr_iter_count += 1
                # calc grad
                if refresh_count > self.opts.refresh_value: refresh_count = 1
                rho = 1 / refresh_count**self.opts.pow
                if refresh_count > 1:
                    grad_curr = self.functions.nl_grads_calc(
                        x = x_list,
                        features = train_features,
                        labels = train_labels
                    )
                    grad_last = self.functions.nl_grads_calc(
                        x = x_list_last,
                        features = train_features,
                        labels = train_labels
                    )
                    grad_list = grad_curr + (1 - rho) * (grad_list - grad_last)
                else:
                    grad_list = self.functions.nl_grads_calc(
                        x = x_list, 
                        features = train_features, 
                        labels = train_labels
                    )
                refresh_count += 1
                # calc kernel
                sq_distance_matrix = (x_list.unsqueeze(0).expand(self.opts.particle_num, self.opts.particle_num, self.model_dim) -\
                    x_list.unsqueeze(1).expand(self.opts.particle_num, self.opts.particle_num, self.model_dim)).pow(2).sum(2)
                kernel_h = self.opts.kernel_h * torch.median(sq_distance_matrix + 1e-5)
                kernel = torch.exp( - sq_distance_matrix/kernel_h)
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
                x_list_last = x_list.detach().clone()
                x_list += v_list * eta + noise_x
                # update v_list
                for i in range(self.opts.particle_num):
                    v_list[i, :] += - self.gamma * eta * v_list[i, :] - eta * self.opts.u * grad_list[i, :] + noise_v[i, :] +\
                        self.opts.weight * eta *\
                        (kernel[:,i].view(-1,1) * (-grad_list - 2*(x_list_last-x_list_last[i,:].view(1,-1))/kernel_h)).mean(0)
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