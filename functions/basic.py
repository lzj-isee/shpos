from abc import ABCMeta, abstractclassmethod

class basic(metaclass=ABCMeta):
    def __init__(self) -> None:
        self.writer = None

    @abstractclassmethod
    def init_net(self, shape):
        pass

    @abstractclassmethod
    def evaluation_mcmc(self, curr_point, curr_iter_count):
        pass

    @abstractclassmethod
    def evaluation_particles(self, sample_list):
        pass

    @abstractclassmethod
    def save_eval_to_tensorboard(self, writer, results, curr_iter_count):
        pass

    @abstractclassmethod
    def save_final_results(self, save_folder, results):
        pass

    @abstractclassmethod
    def set_trainloader(self):
        pass

    @abstractclassmethod
    def nl_grads_calc(self, x, features, labels):
        pass
    