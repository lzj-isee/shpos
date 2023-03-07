import torch
import numpy as np
from tqdm import tqdm
from algorithms.basic import basic

class SVRGPOS(basic):
    def __init__(self, opts):
        algo_name = 'SVRGPOS'
        save_name = algo_name + '_'+\
            'lr[{:.1e}]M[{}]b[{}]k_h[{:.1e}]beta[{:.1e}]s[{}]'.format(\
            opts.lr, opts.particle_num, opts.batch_size,opts.kernel_h, opts.beta, opts.seed)
        super(SVRGPOS,self).__init__(opts, algo_name, save_name)
        print('\nalgorithm and setting: \n',self.save_name)

    def start_sample(self):
        sample_list = self.functions.init_net((self.opts.particle_num, self.model_dim))
        sample_list_snap  = torch.zeros_like(sample_list)
        curr_iter_count = 0
        results = None
        # start updating
        for epoch in tqdm(range(self.opts.epochs)):
            # update per epoch
            sample_list_snap = sample_list.detach().clone()
            grad_alpha = torch.zeros_like(sample_list)
            for _, (train_features, train_labels) in enumerate(self.train_loader_full):
                batchsize = len(train_labels)
                grad_alpha += self.functions.nl_grads_calc(
                    x = sample_list_snap,
                    features = train_features,
                    labels = train_labels
                ) * batchsize / self.train_num
            # inner iter
            for iter, (train_features, train_labels) in enumerate(self.train_loader):
                curr_iter_count += 1
                # calc grad
                grad_curr = self.functions.nl_grads_calc(
                    x = sample_list, 
                    features = train_features, 
                    labels = train_labels
                )
                grad_snap = self.functions.nl_grads_calc(
                    x = sample_list_snap, 
                    features = train_features, 
                    labels = train_labels
                )
                grad = grad_curr - grad_snap + grad_alpha
                # calc kernel
                sq_distance_matrix = (sample_list.unsqueeze(0).expand(self.opts.particle_num, self.opts.particle_num, self.model_dim) -\
                    sample_list.unsqueeze(1).expand(self.opts.particle_num, self.opts.particle_num, self.model_dim)).pow(2).sum(2)
                # update sample list
                kernel_h = self.opts.kernel_h * torch.median(sq_distance_matrix + 1e-5)
                kernel = torch.exp( - sq_distance_matrix/kernel_h)
                for i in range(self.opts.particle_num):
                    sample_list[i, :] = sample_list[i,:] + self.opts.lr * \
                        (kernel[:,i].view(-1, 1)*( - grad - 2*(sample_list - sample_list[i,:].view(1,-1))/kernel_h)).mean(0)
                # evaluation
                if (curr_iter_count - 1) % self.opts.eval_interval == 0:
                    results = self.functions.evaluation_particles(sample_list)
                    self.functions.save_eval_to_tensorboard(
                        self.writer, 
                        results, 
                        (2*epoch+1)*len(self.train_loader)+iter
                    )
                if sample_list.max() != sample_list.max(): raise ValueError('Nan')  # ensurance
        self.functions.save_final_results(self.writer, self.save_folder, results)
        self.post_process()