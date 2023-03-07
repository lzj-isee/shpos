import torch
import numpy as np
from tqdm import tqdm
from algorithms.basic import basic

class WNes(basic):
    def __init__(self, opts):
        algo_name = 'WNes'
        save_name = algo_name + '_'+\
            'lr[{:.1e}]M[{}]b[{}]k_h[{:.1e}]HB[{:.1e}]Sh[{:.2f}]s[{}]'.format(\
            opts.lr, opts.particle_num, opts.batch_size, 
            opts.kernel_h, opts.accHessBnd, opts.accShrink, opts.seed)
        super(WNes,self).__init__(opts, algo_name, save_name)
        print('\nalgorithm and setting: \n',self.save_name)

    def start_sample(self):
        sample_list = self.functions.init_net((self.opts.particle_num, self.model_dim))
        aux_samples = sample_list.detach().clone()
        muH = self.opts.accHessBnd * self.opts.lr
        beta = self.opts.accShrink * np.sqrt(muH)
        c = 1 + beta - 2*(1+beta)*(2+beta)*muH / \
            (np.sqrt(beta**2+4*(1+beta)*muH)-beta+2*(1+beta)*muH)
        curr_iter_count = 0
        results = None
        # start updating
        for epoch in tqdm(range(self.opts.epochs)):
            for iter, (train_features, train_labels) in enumerate(self.train_loader):
                curr_iter_count += 1
                # calc grad
                grad_list = self.functions.nl_grads_calc(
                    x = sample_list, 
                    features = train_features, 
                    labels = train_labels
                )
                # calc kernel
                sq_distance_matrix = (sample_list.unsqueeze(0).expand(self.opts.particle_num, self.opts.particle_num, self.model_dim) -\
                    sample_list.unsqueeze(1).expand(self.opts.particle_num, self.opts.particle_num, self.model_dim)).pow(2).sum(2)
                # update sample list
                kernel_h = self.opts.kernel_h * torch.median(sq_distance_matrix + 1e-5)
                kernel = torch.exp( - sq_distance_matrix/kernel_h)
                for i in range(self.opts.particle_num):
                    old_sample_list = sample_list.detach().clone()
                    sample_list[i, :] = aux_samples[i, :] + self.opts.lr * \
                        (kernel[:,i].view(-1, 1)*( - grad_list - 2*(sample_list - sample_list[i, :].view(1,-1))/kernel_h)).mean(0)
                    aux_samples[i, :] = sample_list[i, :] + c * (sample_list[i, :] - old_sample_list[i, :])
                # evaluation
                if (curr_iter_count - 1) % self.opts.eval_interval == 0:
                    results = self.functions.evaluation_particles(aux_samples)  # ref AWGF, we use the aux_samples
                    self.functions.save_eval_to_tensorboard(
                        self.writer, 
                        results, 
                        epoch*len(self.train_loader)+iter
                    )
                if sample_list.max() != sample_list.max(): raise ValueError('Nan')  # ensurance
        self.functions.save_final_results(self.writer, self.save_folder, results)
        self.post_process()