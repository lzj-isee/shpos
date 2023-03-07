from functions.basic import basic
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
# for toy_dataset (toy_task)

class functions(basic):
    def __init__(self, opts, device):
        self.opts = opts
        self.device = device
        self.model_dim = opts.dim
        self.cov_matrix = opts.cov * torch.ones((self.model_dim, self.model_dim)).to(self.device)
        self.cov_matrix = self.cov_matrix - torch.diag(torch.diag(self.cov_matrix)) + \
            torch.diag(opts.var * torch.ones(self.model_dim).to(self.device))
        self.in_cov_matrix = torch.inverse(self.cov_matrix)
        self.in_cov_matrix_2dim = torch.inverse(self.cov_matrix[0:2,0:2])
        self.mean = opts.mean * torch.ones((1, self.model_dim)).to(self.device)
        self.ksd_value = None
        self.w2_value = None
        self.mse_value = None
        self.cov_error_value = None
        p = np.random.multivariate_normal(
            mean = opts.mean * np.ones(self.model_dim), 
            cov = self.cov_matrix.cpu().numpy(), 
            size = self.opts.particle_num
        )
        self.ref_samples = torch.Tensor(p).to(self.device)
        self.init_mu = opts.init_mean * torch.ones(self.model_dim).to(self.device)
        self.save_samples = []

    def init_net(self, shape):
        result = torch.randn(shape, device = self.device) * self.opts.init_std + self.init_mu
        return result

    def set_trainloader(self):  # we do not use this function in toydataset task
        self.train_loader = [[0, 0]]

    @torch.no_grad()
    def pdf_calc(self, x):  # x: M * dim matrix
        result = torch.exp( - (torch.matmul((x-self.mean), self.in_cov_matrix) * (x-self.mean)).sum(1) / 2)
        return result   # M array

    @torch.no_grad()
    def nl_grads_calc(self, x, features = None, labels = None): # x: M * dim matrix
        result = torch.matmul((x-self.mean), self.in_cov_matrix)
        return result   # M*dim matrix

    #----------------------------------------------------------------------------------------------------
    # Evaluation code, work cooperatively
    #----------------------------------------------------------------------------------------------------

    def evaluate_ksd(self, sample_list):
        grad_list = self.nl_grads_calc(
            x = sample_list, 
            features = None, 
            labels = None
        )
        matrix_sub = sample_list.unsqueeze(0).expand(self.opts.particle_num, self.opts.particle_num, self.model_dim) -\
            sample_list.unsqueeze(1).expand(self.opts.particle_num, self.opts.particle_num, self.model_dim)
        # calc kernel
        sq_distance_matrix = matrix_sub.pow(2).sum(2)
        # update sample list
        if self.opts.ksd_kernel == -1:
            kernel_h = 2 * torch.median(sq_distance_matrix + 1e-5)  # bandwidth choosen to be the meadian of the data distances
        else:
            kernel_h = self.opts.ksd_kernel
        kernel = torch.exp( - sq_distance_matrix/kernel_h)
        trace_nabla = (( - matrix_sub.pow(2) * 4 / kernel_h**2 + (2 / kernel_h) ) * kernel.unsqueeze(2)).sum(2)
        part_1 = torch.matmul(grad_list, grad_list.T) * kernel
        part_2 = ((2 * matrix_sub / kernel_h * kernel.unsqueeze(2)) * grad_list).sum(2)
        part_3 = ((2 * matrix_sub / kernel_h * kernel.unsqueeze(2)) * grad_list.unsqueeze(1)).sum(2)
        # we use U-statistics here
        ksd_values_matrix = part_1 + part_2 + part_3 + trace_nabla
        self.ksd_value = (ksd_values_matrix.sum() - torch.diag(ksd_values_matrix).sum()) / \
            self.opts.particle_num / (self.opts.particle_num - 1)

    def sinkhorn_loss(self, x, y):
        cost_matrix = (x.unsqueeze(0).expand(self.opts.particle_num, self.opts.particle_num, self.model_dim) -\
            y.unsqueeze(1).expand(self.opts.particle_num, self.opts.particle_num, self.model_dim)).pow(2).sum(2)
        mu = torch.ones(self.opts.particle_num, device = self.device) / self.opts.particle_num
        nu = torch.ones(self.opts.particle_num, device = self.device) / self.opts.particle_num
        def M(u,v):
            "Modified cost for logarithmic updates"
            "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
            return ( - cost_matrix + u.unsqueeze(1) + v.unsqueeze(0)) / self.opts.w2_epsilon
        def lse(A):
            "log-sum-exp"
            return torch.log(torch.exp(A).sum(1, keepdim=True) + 1e-6)  # add 10^-6 to prevent NaN
        u, v, err = 0. * mu, 0. * nu, 0.
        actual_nits = 0  # to check if algorithm terminates because of threshold or max iterations reached
        for i in range(self.opts.w2_max_iter):
            u1 = u
            u = self.opts.w2_epsilon * (torch.log(mu) - lse(M(u, v)).squeeze()) + u
            v = self.opts.w2_epsilon * (torch.log(nu) - lse(M(u, v).t()).squeeze()) + v
            err = (u - u1).abs().sum()
            actual_nits += 1
            if (err < self.opts.w2_thresh).cpu().numpy():
                break
        U, V = u, v
        pi = torch.exp(M(U, V))  # Transport plan pi = diag(a)*K*diag(b)
        cost = torch.sum(pi * cost_matrix)  # Sinkhorn cost
        return cost

    def evaluation_particles(self, sample_list):
        # evaluate w2
        self.w2_value = 2 * self.sinkhorn_loss(sample_list, self.ref_samples) - \
            self.sinkhorn_loss(sample_list, sample_list) - self.sinkhorn_loss(self.ref_samples, self.ref_samples)
        # eval KSD
        self.evaluate_ksd(sample_list)
        # eval MSE
        self.mse_value = (sample_list-self.mean).mean(0).pow(2).sum()
        # eval cov error
        mean = sample_list.mean(0)
        cov = torch.matmul((sample_list - mean).T, (sample_list - mean)) / (self.opts.particle_num - 1)
        self.cov_error_value = (cov - self.cov_matrix).abs().mean()
        # save samples
        self.save_samples.append(sample_list.cpu().numpy())
        return sample_list

    def evaluation_mcmc(self, curr_point, curr_iter_count):
        pass

    @torch.no_grad()
    def pdf_calc_2dim(self, x):  # x: M * dim matrix
        result = torch.exp(
            - (torch.matmul((x-self.mean[:,0:2]), self.in_cov_matrix_2dim) * (x-self.mean[:,0:2])).sum(1) / 2)
        return result   # M array

    def save_eval_to_tensorboard(self, writer, results, curr_iter_count):
        samples = results.cpu().numpy()
        #samples = self.ref_samples.cpu().numpy()   # debug
        writer.add_scalar('W2', self.w2_value.item(), curr_iter_count)
        writer.add_scalar('KSD', self.ksd_value.item(), curr_iter_count)
        writer.add_scalar('MSE', self.mse_value.item(), curr_iter_count)
        writer.add_scalar('Cov_Error', self.cov_error_value.item(), curr_iter_count)
        # calc mesh
        x, y = torch.linspace(-6, 6, 121, device = self.device), torch.linspace(-6, 6, 121, device = self.device)
        grid_X, grid_Y = torch.meshgrid(x,y)
        loc = torch.cat([grid_X.unsqueeze(2), grid_Y.unsqueeze(2)], dim = 2)
        num = len(loc)
        pdf = self.pdf_calc_2dim(loc.view(-1, 2))
        pdf = pdf.view(num, num)
        fig = plt.figure(num = 1)
        plt.contour(grid_X.cpu().numpy(), grid_Y.cpu().numpy(), pdf.cpu().numpy())
        plt.scatter(samples[:, 0], samples[:, 1],alpha = 0.5)
        plt.ylim([ - self.opts.ylim, self.opts.ylim])
        plt.xlim([ - self.opts.xlim, self.opts.xlim])
        writer.add_figure(tag = 'samples', figure = fig, global_step = curr_iter_count)
        plt.close()

    def save_final_results(self, writer, save_folder, results):
        samples = np.array(self.save_samples)
        if self.opts.save_samples == True:
            np.save(os.path.join(save_folder, 'samples.npy'), samples)