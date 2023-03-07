import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import os
import pandas as pd
from libsvm.python.svmutil import *
from sklearn.model_selection import train_test_split
from functions import save_final_results

# Bayesian neural networks for regression 
# !!! we consider the situation of mutiple outputs in regression task
# !!! check the log_prior vairable, the additional +log_gamma and +log_lambda

class functions(object):
    def __init__(self, opts, device):
        self.device = device
        self.opts = opts
        self.n_hidden = opts.n_hidden
        self.param_a = opts.gamma_a
        self.param_b = opts.gamma_b

    def set_trainloader(self):
        train_set = TensorDataset(self.train_features, self.train_labels)
        self.train_loader = DataLoader(
            dataset = train_set,
            batch_size = self.opts.batch_size,
            shuffle = True,
            drop_last = True
        )
        self.train_loader_full = DataLoader(
            dataset = train_set,
            batch_size = self.opts.pre_batch_size,
            shuffle = False,
            drop_last = False
        )

    def pack_params(self, w1, b1, w2, b2, log_gamma, log_lambda):
        params = torch.cat([w1.flatten(), b1, w2.flatten(), b2, log_gamma.view(-1), log_lambda.view(-1)])
        return params
    
    def unpack_params(self, params):
        w1 = params[:self.data_dim * self.n_hidden].view(self.data_dim, self.n_hidden)
        b1 = params[self.data_dim * self.n_hidden : (self.data_dim + 1) * self.n_hidden]
        temp = params[(self.data_dim + 1) * self.n_hidden:]
        w2, b2 = temp[:self.n_hidden * self.out_dim].view(self.n_hidden, self.out_dim), temp[-2-self.out_dim:-2]
        # the last two parameters are log variance
        log_gamma, log_lambda = temp[-2], temp[-1]
        return w1, b1, w2, b2, log_gamma, log_lambda

    def init_params(self):
        w1 = 1.0 / np.sqrt(self.data_dim + 1) * torch.randn((self.data_dim, self.n_hidden), device = self.device)
        b1 = torch.zeros((self.n_hidden, ), device = self.device)
        w2 = 1.0 / np.sqrt(self.n_hidden + 1) * torch.randn((self.n_hidden, self.out_dim), device = self.device)
        b2 = torch.zeros((self.out_dim, ), device = self.device)
        log_gamma = torch.log(torch.ones((1, ), device = self.device)*np.random.gamma(self.param_a, self.param_b))
        log_lambda = torch.log(torch.ones((1, ), device = self.device)*np.random.gamma(self.param_a, self.param_b))
        return w1, b1, w2, b2, log_gamma, log_lambda

    @torch.no_grad()
    def init_net(self, shape):
        param_group = torch.zeros(shape, device = self.device)
        if len(param_group.shape) == 1: # single particle
            param_group = param_group.view(1, -1)
        if len(param_group.shape) == 2:   # multiple particles
            for i in range(len(param_group)):
                w1, b1, w2, b2, log_gamma, log_lambda = self.init_params()
                # use better initiaization for gamma
                index = torch.multinomial(1/torch.ones(self.train_num, device = self.device), \
                np.min([self.train_num, 1000]), replacement = False)
                y_predic = self.prediction(
                    self.pack_params(w1, b1, w2, b2, log_gamma, log_lambda), 
                    self.train_features[index]
                )
                log_gamma = - torch.log((y_predic - self.train_labels[index].view(-1, self.out_dim)).pow(2).mean())
                param_group[i, :] = self.pack_params(w1, b1, w2, b2, log_gamma, log_lambda)
        else:
            raise ValueError('invalid particle dim')
        if len(shape) == 1:
            param_group = param_group.squeeze()
        return param_group # recover the dim for single particle
    
    #-----------------------------------------------------------------------------------------------------------------------
    #   Main process code
    #-----------------------------------------------------------------------------------------------------------------------

    def prediction(self, x, features):  # accomplish this function using loop, may be slow.
        if len(x.shape) == 1:   # single particle, and we do not recover dimemsion in this function
            x = x.view(1, -1)
        predicts = torch.zeros((len(x), len(features), self.out_dim), device = self.device)
        for i in range(len(x)):
            w1, b1, w2, b2, _, _ = self.unpack_params(x[i, :])
            outputs = torch.matmul(torch.relu(torch.matmul(features, w1) + b1), w2) + b2
            predicts[i,:,:] = outputs.view(-1, self.out_dim)
        return predicts # M*batch_size*dim matrix
    
    def prediction_parallel(self, x, features): # accomplish prediction parallelly, should be efficient
        if len(x.shape) == 1:   # single particle, and we do not recover dimension in this function
            x = x.view(1, -1)
        num_particles = len(x)
        w1_s = x[:, :self.data_dim * self.n_hidden].view(num_particles, self.data_dim, self.n_hidden)
        b1_s = x[:, self.data_dim * self.n_hidden : (self.data_dim + 1) * self.n_hidden].unsqueeze(1)
        temp = x[:, (self.data_dim + 1) * self.n_hidden:]
        w2_s = temp[:, :self.n_hidden * self.out_dim].view(num_particles, self.n_hidden, self.out_dim)
        b2_s = temp[:, -2 - self.out_dim : -2].unsqueeze(1)
        log_gamma, log_lambda = temp[:, -2], temp[:, -1]    # note the dimension of this variable
        outputs = torch.matmul(torch.relu(torch.matmul(features, w1_s) + b1_s), w2_s) + b2_s
        return outputs, [w1_s, b1_s, w2_s, b2_s, log_gamma, log_lambda] # return the params for the following calc
    
    def log_posteriors_calc(self, x, features, labels): # accomplish this function using loop, may be slow.
        if len(x.shape) == 1:   # single particle, and we do not need to recover dimension here
            x = x.view(1, -1)
        log_posteriors = torch.zeros(len(x), device = self.device)
        predicts = self.prediction(x, features)
        batch_size = len(features)
        for i in range(len(x)):
            predict = predicts[i].squeeze()
            w1, b1, w2, b2, log_gamma, log_lambda = self.unpack_params(x[i,:])
            log_likeli_data = -0.5 * batch_size * (np.log(2*np.pi) - log_gamma) - (log_gamma.exp()/2) * (predict - labels).pow(2).sum()
            log_prior_data = (self.param_a - 1) * log_gamma - self.param_b * log_gamma.exp() + log_gamma  # NOTE: why additional +log_gamma?
            log_prior_w = -0.5 * (self.model_dim - 2) * (np.log(2*np.pi) - log_gamma) - (log_gamma.exp()/2) * \
                (w1.pow(2).sum() + w2.pow(2).sum() + b1.pow(2).sum() + b2.pow(2).sum()) + \
                (self.param_a - 1) * log_lambda - self.param_b * log_lambda.exp() + log_lambda  # NOTE: why additional +log_lambda?
            log_posteriors[i] = log_likeli_data * self.train_num / batch_size + log_prior_data + log_prior_w
        return log_posteriors

    def log_posteriors_calc_parallel(self, x, features, labels):    # accomplish this function parallelly, shoue be efficient
        if len(x.shape) == 1:
            x = x.view(1, -1)
        batch_size = len(features)
        predictions, unpack_params = self.prediction_parallel(x, features)
        w1_s, b1_s, w2_s, b2_s, log_gamma, log_lambda = unpack_params
        log_likeli_data = -0.5 * batch_size * (np.log(2*np.pi) - log_gamma) - (log_gamma.exp()/2) * \
            (predictions - labels.view(-1, self.out_dim)).pow(2).sum((1,2))
        log_prior_data = (self.param_a - 1) * log_gamma - self.param_b * log_gamma.exp() + log_gamma # NOTE: +log_gamma
        log_prior_w = -0.5 * (self.model_dim - 2) * (np.log(2*np.pi) - log_gamma) - (log_gamma.exp()/2) * \
            (w1_s.pow(2).sum((1,2))+w2_s.pow(2).sum((1,2))+b1_s.pow(2).sum((1,2))+b2_s.pow(2).sum((1,2))) + \
            (self.param_a - 1) * log_lambda - self.param_b * log_lambda.exp() + log_lambda  # NOTE: +log_lambda
        log_posteriors = log_likeli_data * self.train_num / batch_size + log_prior_data + log_prior_w
        return log_posteriors

    def nl_grads_calc(self, x, features, labels):
        if len(x.shape) == 1:   # single particle
            particles = x.view(1, -1).detach()#.clone().requires_grad_(True)
        else:
            particles = x.detach()#.clone().requires_grad_(True)
        particles = particles.requires_grad_(True)
        # NOTE: 2020/12/5, we calc the forward process in a better way (maybe)
        loss = - self.log_posteriors_calc_parallel(particles, features, labels)
        #loss = - self.log_posteriors_calc(particles, features, labels)
        grad_list = torch.autograd.grad(
            outputs = loss.sum(),
            inputs = particles
        )[0]
        if len(x.shape) == 1:
            return grad_list.squeeze()  # recover the dim
        else:
            return grad_list

    def save_final_results(self,writer, save_folder, result_dict):
        save_final_results(save_folder, result_dict)

    def save_eval_to_tensorboard(self, writer, results, global_step):
        writer.add_scalar('train nll', results['train_nll'], global_step = global_step)
        writer.add_scalar('train rmse', results['train_rmse'], global_step = global_step)
        writer.add_scalar('test nll', results['test_nll'], global_step = global_step)
        writer.add_scalar('test rmse', results['test_rmse'], global_step = global_step)

    #-------------------------------------------------------------------------------------------------------------
    #   Evaluation code
    #-------------------------------------------------------------------------------------------------------------

    @torch.no_grad()
    def likelihoods_predicts_calc(self, particles, features, labels, is_Test = False):
        if len(particles.shape) == 1:   # single particle
            particles = particles.view(1, -1)
        predicts, _ = self.prediction_parallel(particles, features)
        if is_Test: predicts = predicts * self.std_labels + self.mean_labels
        log_gamma_s = particles[:, -2].view(-1, 1)
        part_1 = log_gamma_s.exp().sqrt() / np.sqrt(2 * np.pi)
        part_2 = torch.exp(- (predicts - labels.view(-1, self.out_dim)).pow(2).sum(2) / 2 * log_gamma_s.exp())
        likelihoods = part_1 * part_2 + 1e-5
        return likelihoods, predicts
    
    @torch.no_grad()
    def _outputs_to_evaluations(self, train_likeli, train_pred, test_likeli, test_pred):
        train_predict_avg, test_predict_avg = train_pred.mean(0), test_pred.mean(0)
        train_likeli_avg, test_likeli_avg = train_likeli.mean(0), test_likeli.mean(0)
        train_nll, test_nll = - train_likeli_avg.log().mean(), - test_likeli_avg.log().mean()
        train_rmse = (train_predict_avg.squeeze() - self.train_labels).pow(2).mean(0).sum().sqrt()
        test_rmse = (test_predict_avg.squeeze() - self.test_labels).pow(2).mean(0).sum().sqrt()
        return train_nll, train_rmse, test_nll, test_rmse

    @torch.no_grad()
    def evaluation_particles(self, particles):
        if len(particles.shape) == 1:   # single particle
            particles = particles.view(1, -1)
        train_likeli, train_predicts = self.likelihoods_predicts_calc(particles, self.train_features, self.train_labels, is_Test=False)
        test_likeli, test_predicts = self.likelihoods_predicts_calc(particles, self.test_features, self.test_labels, is_Test=True)
        train_nll, train_rmse, test_nll, test_rmse = self._outputs_to_evaluations(
            train_likeli = train_likeli,
            train_pred = train_predicts,
            test_likeli = test_likeli,
            test_pred = test_predicts
        )
        return {'train_nll':train_nll.item(), 'train_rmse':train_rmse.item(),\
            'test_nll':test_nll.item(), 'test_rmse':test_rmse.item()}
    
    @torch.no_grad()
    def evaluation_mcmc(self, x, curr_iter_count):  #NOTE: have not been tested
        train_likeli, train_predicts = self.likelihoods_predicts_calc(x, self.train_features, self.train_labels)
        test_likeli, test_predicts = self.likelihoods_predicts_calc(x, self.test_features, self.test_labels)
        if curr_iter_count > self.opts.burn_in:
            self.train_outputs_avg = ((curr_iter_count - self.opts.burn_in - 1) * self.train_outputs_avg + train_predicts)/\
                (curr_iter_count - self.opts.burn_in)
            self.test_outputs_avg = ((curr_iter_count - self.opts.burn_in - 1) * self.test_outputs_avg + train_predicts)/\
                (curr_iter_count - self.opts.burn_in)
            train_nll, train_rmse, test_nll, test_rmse = self._outputs_to_evaluations(
                train_likeli = train_likeli,
                train_pred = self.train_outputs_avg,
                test_likelit = test_likeli,
                test_pred = self.test_outputs_avg
            )
        else:
            # no average
            train_nll, train_rmse, test_nll, test_rmse = self._outputs_to_evaluations(
            train_likeli = train_likeli,
            train_pred = train_predicts,
            test_likeli = test_likeli,
            test_pred = test_predicts
        )
        return {'train_nll':train_nll.item(), 'train_rmse':train_rmse.item(),\
            'test_nll':test_nll.item(), 'test_rmse':test_rmse.item()}

    #--------------------------------------------------------------------------------------------------------------
    # Code of loading dataset
    #--------------------------------------------------------------------------------------------------------------
    
    def load_dataset(self, dataset_name):
        main_folder = os.path.join('datasets')
        self.out_dim = 1
        # load and split dataset
        if dataset_name in 'YearPredictionMSD':  # no need to split dataset
            train_path = os.path.join(main_folder, dataset_name, dataset_name + '-train.txt')
            test_path = os.path.join(main_folder, dataset_name, dataset_name + '-test.txt')
            train_labels_raw, train_features_raw = svm_read_problem(train_path, return_scipy = True)
            test_labels_raw, test_features_raw = svm_read_problem(test_path, return_scipy = True)
            train_features_raw = train_features_raw.toarray()
            test_features_raw = test_features_raw.toarray()
        elif dataset_name in 'abalone, boston, mpg, cpusmall, cadata, space':    # split dataset
            data_path = os.path.join(main_folder, dataset_name, dataset_name + '.txt')
            labels_raw, features_raw = svm_read_problem(data_path, return_scipy = True)
            train_features_raw, test_features_raw, train_labels_raw, test_labels_raw = train_test_split(
                features_raw, labels_raw, test_size = self.opts.test_size, random_state = self.opts.split_seed)
            train_features_raw = train_features_raw.toarray()
            test_features_raw = test_features_raw.toarray()
        elif dataset_name in 'concrete':    # load xls file and split the dataset
            data_path = os.path.join(main_folder, dataset_name, dataset_name + '.xls')
            data_raw = pd.read_excel(data_path, header = 0).values
            labels_raw, features_raw = data_raw[:, -1], data_raw[:,:-1]
            train_features_raw, test_features_raw, train_labels_raw, test_labels_raw = train_test_split(
                features_raw, labels_raw, test_size = self.opts.test_size, random_state = self.opts.split_seed)
        elif dataset_name in 'energy, kin8nm, casp, superconduct, slice, online, sgemm, electrical':  # load csv file and split the dataset
            data_path = os.path.join(main_folder, dataset_name, dataset_name + '.csv')
            data_raw = pd.read_csv(data_path, header = 0).values
            if dataset_name in 'energy':
                labels_raw, features_raw = data_raw[:, 1].astype(np.float32), data_raw[:, 2:].astype(np.float32)
            elif dataset_name in 'kin8nm, superconduct, slice':
                labels_raw, features_raw = data_raw[:, -1], data_raw[:, :-1]
            elif dataset_name in 'casp':
                labels_raw, features_raw = data_raw[:, 0], data_raw[:, 1:]
            elif dataset_name in 'online':
                labels_raw, features_raw = data_raw[:, -1].astype(np.float32), data_raw[:, 1:-1].astype(np.float32)
            elif dataset_name in 'sgemm':
                labels_raw, features_raw = data_raw[:, -4], data_raw[:, :-4]
            elif dataset_name in 'electrical':
                labels_raw, features_raw = data_raw[:, -2].astype(np.float32), data_raw[:, :-2].astype(np.float32)
            else:
                raise ValueError('dataset {} not found'.format(dataset_name))
            train_features_raw, test_features_raw, train_labels_raw, test_labels_raw = train_test_split(
                features_raw, labels_raw, test_size = self.opts.test_size, random_state = self.opts.split_seed)    
        elif dataset_name in 'WineRed, WineWhite':
            data_path = os.path.join(main_folder, dataset_name, dataset_name + '.csv')
            attris = pd.read_csv(data_path, header = 0).values.reshape(-1).tolist()
            data_raw = []
            for attr in attris:
                data_raw.append([eval(number) for number in attr.split(';')])
            data_raw = np.array(data_raw)
            labels_raw, features_raw = data_raw[:, -1], data_raw[:,:-1]
            train_features_raw, test_features_raw, train_labels_raw, test_labels_raw = train_test_split(
                features_raw, labels_raw, test_size = self.opts.test_size, random_state = self.opts.split_seed)  
        else:   # TODO:  naval, WineRed, WineWhite
            raise ValueError('dataset {} not found'.format(dataset_name))
        # to Tensor
        self.train_features = torch.from_numpy(train_features_raw).float().to(self.device)
        self.train_labels = torch.from_numpy(train_labels_raw).float().to(self.device)
        self.test_features = torch.from_numpy(test_features_raw).float().to(self.device)
        self.test_labels = torch.from_numpy(test_labels_raw).float().to(self.device)
        # Normalization
        self.std_features = torch.std(self.train_features, dim = 0, unbiased = True)
        self.std_features[self.std_features == 0] = 1
        self.mean_features = torch.mean(self.train_features, dim = 0)
        self.std_labels = torch.std(self.train_labels, dim = 0)
        self.mean_labels = torch.mean(self.train_labels, dim = 0)
        self.train_features = (self.train_features - self.mean_features) / self.std_features
        self.train_labels = (self.train_labels - self.mean_labels) / self.std_labels
        self.test_features = (self.test_features - self.mean_features) / self.std_features
        #self.test_labels = (self.test_labels - self.mean_labels) / self.std_labels
        # record information
        self.data_dim = len(self.train_features[0])
        self.train_num = len(self.train_features)
        self.test_num = len(self.test_features)
        self.model_dim = self.data_dim * self.n_hidden + self.n_hidden * self.out_dim +\
            self.n_hidden + self.out_dim + 2  # 2 variances
        # creat eval tensor for SGLD...
        self.train_outputs_avg = torch.zeros_like(self.train_labels)
        self.test_outputs_avg = torch.zeros_like(self.test_labels)

