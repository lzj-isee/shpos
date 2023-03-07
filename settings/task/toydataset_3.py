from functions import str2bool
class args(object):
    def __init__(self, parser):
        self.parser = parser
        parser.add_argument('--dim', type = int, default = 10)
        parser.add_argument('--var', type = float, default = 6)
        parser.add_argument('--cov', type = float, default =  0.9 * 6)
        parser.add_argument('--mean', type = float, default = 0)
        parser.add_argument('--ksd_kernel', type = float, default = 1.5)
        parser.add_argument('--w2_epsilon', type = float, default = 0.1)
        parser.add_argument('--w2_max_iter', type = int, default = 100)
        parser.add_argument('--w2_thresh', type = float, default = 0.1)
        parser.add_argument('--init_mean', type = float, default = -4)
        parser.add_argument('--init_std', type = float, default = 0.25)
        parser.add_argument('--xlim', type = float, default = 6)
        parser.add_argument('--ylim', type = float, default = 6)
        parser.add_argument('--save_samples', type = str2bool, default = True)

    def return_parser(self):
        return self.parser