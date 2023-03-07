class args(object):
    def __init__(self, parser):
        self.parser = parser
        parser.add_argument('--iters', type = int, default = 100)
        parser.add_argument('--particle_num', type = int, default = 1000)
        parser.add_argument('--batch_size', type = int, default = 100)
        parser.add_argument('--pre_batch_size', type = int, default = 1000)
        # parser.add_argument('--init_std', type = float, default = 0.001)
        # kernel_h = 1 / ln(particle_num)
        parser.add_argument('--kernel_h', type = float, default = 0.14)
        # weight = 1 / beta
        parser.add_argument('--beta', type = float, default = 100)
        parser.add_argument('--lr', type=float, default = 1e-4)
    
    def return_parser(self):
        return self.parser