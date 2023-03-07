import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import os
from libsvm.python.svmutil import *
from sklearn.model_selection import train_test_split
from functions import save_final_results

# for logistic regression task

class functions(object):
    def __init__(self, opts, device):
        self.device = device
        self.opts = opts

    def init_net(self, shape):
        return torch.randn(shape, device = self.device) * self.opts.init_std

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

    @torch.no_grad()
    def likelihood_calc(self, x, features, labels): # able to calc parallelly
        if len(x.shape) == 1:   # check the dimension
            models = x.view(1, -1)
        else:
            models = x  # M*D matrix
        Z = torch.matmul(features, models.T) * labels.view(-1, 1) # B*M matrix
        result = torch.sigmoid(Z).T # M*B matrix
        if len(x.shape) == 1:   # recover the dimension
            result = result.view(-1)
        return result

    @torch.no_grad()
    def likelihood_calc_sigle(self, x, features, labels):   # only for data dim = 1
        Z = torch.matmul(features,x)*labels
        result = torch.sigmoid(Z)
        return result

    @torch.no_grad()
    def nl_grads_calc(self, x, features, labels):   # able to calc parallelly
        if len(x.shape) == 1:   # check the dimension
            models = x.view(1, -1)
        else:
            models = x  # M*D matrix
        batch_size = len(features)
        Z = - torch.matmul(features, models.T) * labels.view(-1, 1) # B*M matrix
        A = torch.sigmoid(Z)    # B*M matrix
        temp = - (A * labels.view(-1, 1)).T   # M*B matrix
        grad = torch.matmul(temp, features) * (self.train_num / batch_size) + self.opts.weight_decay * models
        if len(x.shape) == 1:    # recover the dimension
            grad = grad.view(-1)
        return grad

    @torch.no_grad()
    def nl_grads_calc_single(self, x, features, labels): # only for data dim = 1
        # Calculate the gradients
        Z = -torch.matmul(features,x)*labels
        A = torch.sigmoid(Z)
        grad = ( - (A * labels).view(-1,1) * features).mean(0) * self.train_num + self.opts.weight_decay * x
        return grad

    def save_final_results(self,writer, save_folder, result_dict):
        save_final_results(save_folder, result_dict)

    def save_eval_to_tensorboard(self, writer, results, global_step):
        writer.add_scalar('train nll', results['train_nll'], global_step = global_step)
        writer.add_scalar('train error', results['train_error'], global_step = global_step)
        writer.add_scalar('test nll', results['test_nll'], global_step = global_step)
        writer.add_scalar('test error', results['test_error'], global_step = global_step)

    @torch.no_grad()
    def _outputs_to_evaluations(self, train_outputs, test_outputs):
        train_nll = ( - torch.log(train_outputs)).mean()
        train_error = 1 - ((torch.round(train_outputs)).sum() / self.train_num)
        test_nll = ( - torch.log(test_outputs)).mean()
        test_error = 1 - ((torch.round(test_outputs)).sum() / self.test_num)
        return train_nll, train_error, test_nll, test_error
        
    @torch.no_grad()
    def evaluation_particles(self, particles):
        if len(particles.shape) == 1:
            x_s = particles.view(1, -1)
        else:
            x_s = particles
        train_outputs_avg = self.likelihood_calc(x_s, self.train_features, self.train_labels).mean(0)
        test_outputs_avg = self.likelihood_calc(x_s, self.test_features, self.test_labels).mean(0)
        train_nll, train_error, test_nll, test_error = self._outputs_to_evaluations(train_outputs_avg, test_outputs_avg)
        return {'train_nll':train_nll.item(), 'train_error':train_error.item(),\
            'test_nll':test_nll.item(), 'test_error':test_error.item()}

    @torch.no_grad()
    def evaluation_mcmc(self, x, curr_iter_count):
        train_outputs = self.likelihood_calc_sigle(x, self.train_features, self.train_labels)
        test_outputs = self.likelihood_calc_sigle(x, self.test_features, self.test_labels)
        if curr_iter_count > self.opts.burn_in:
            self.train_outputs_avg = ((curr_iter_count - self.opts.burn_in - 1) * self.train_outputs_avg + train_outputs)/\
                (curr_iter_count - self.opts.burn_in)
            self.test_outputs_avg = ((curr_iter_count - self.opts.burn_in - 1) * self.test_outputs_avg + test_outputs)/\
                (curr_iter_count - self.opts.burn_in)
            train_nll, train_error, test_nll, test_error = self._outputs_to_evaluations(self.train_outputs_avg, self.test_outputs_avg)
        else:
            # no average
            train_nll, train_error, test_nll, test_error = self._outputs_to_evaluations(train_outputs, test_outputs)
        return {'train_nll':train_nll.item(), 'train_error':train_error.item(),\
            'test_nll':test_nll.item(), 'test_error':test_error.item()}


    def load_dataset(self, dataset_name):
        main_folder = os.path.join('datasets')
        # load and split dataset
        if dataset_name in 'a3a, a9a, ijcnn, gisette, w8a, a8a, codrna, madelon':  # no need to split dataset
            train_path = os.path.join(main_folder, dataset_name, dataset_name + '-train.txt')
            test_path = os.path.join(main_folder, dataset_name, dataset_name + '-test.txt')
            train_labels_raw, train_features_raw = svm_read_problem(train_path, return_scipy = True)
            test_labels_raw, test_features_raw = svm_read_problem(test_path, return_scipy = True)
        elif dataset_name in 'mushrooms, pima, covtype, phishing, susy':    # split dataset
            data_path = os.path.join(main_folder, dataset_name, dataset_name + '.txt')
            labels_raw, features_raw = svm_read_problem(data_path, return_scipy = True)
            if dataset_name in 'mushrooms, covtype': labels_raw = (labels_raw - 1.5) * 2
            if dataset_name in 'phishing, susy': labels_raw = (labels_raw - 0.5) * 2
            train_features_raw, test_features_raw, train_labels_raw, test_labels_raw = train_test_split(
                features_raw, labels_raw, test_size = self.opts.test_size, random_state = self.opts.split_seed)
        else:
            raise ValueError('dataset {} not found'.format(dataset_name))
        # some extra process for certain dataset
        train_features_raw = train_features_raw.toarray()
        test_features_raw = test_features_raw.toarray()
        self.train_num = len(train_labels_raw)
        self.test_num = len(test_labels_raw)
        if dataset_name == 'a3a':
            train_features_raw = np.concatenate((train_features_raw, np.zeros([self.train_num,1])), axis=1)
        if dataset_name in 'a9a, a8a':
            test_features_raw = np.concatenate((test_features_raw, np.zeros([self.test_num,1])), axis=1)
        # concatenate bias
        train_features_raw = np.concatenate((np.ones([self.train_num,1]), train_features_raw), axis=1)
        test_features_raw = np.concatenate((np.ones([self.test_num,1]), test_features_raw), axis=1)
        # to Tensor
        self.train_features = torch.from_numpy(train_features_raw).float().to(self.device)
        self.train_labels = torch.from_numpy(train_labels_raw).float().to(self.device)
        self.test_features = torch.from_numpy(test_features_raw).float().to(self.device)
        self.test_labels = torch.from_numpy(test_labels_raw).float().to(self.device)
        self.data_dim = len(self.train_features[0])
        self.model_dim = self.data_dim
        # creat eval tensor
        self.train_outputs_avg = torch.zeros(self.train_num, device=self.device)
        self.test_outputs_avg = torch.zeros(self.test_num, device=self.device)
        # extra process for dataset of tensor
        if dataset_name in 'madelon, codrna, covtype':
            self.std_features = torch.std(self.train_features, dim = 0, unbiased = True)
            self.std_features[self.std_features == 0] = 1
            self.mean_features = torch.mean(self.train_features, dim = 0)
            self.mean_features[0] = 0
            self.train_features = (self.train_features - self.mean_features) / self.std_features
            self.test_features = (self.test_features - self.mean_features) / self.std_features

