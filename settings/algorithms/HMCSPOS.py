class args(object):
    def __init__(self, parser):
        self.parser = parser
        parser.add_argument('--iters', type = int, default = 101)
        parser.add_argument('--particle_num', type = int, default = 1000)
        parser.add_argument('--batch_size', type = int, default = 100)
        # parser.add_argument('--init_std', type = float, default = 0.001)
        parser.add_argument('--lr', type=float, default = 0.06)
        # kernel_h = 1 / ln(particle_num)
        parser.add_argument('--kernel_h', type = float, default = 0.14)
        # weight = 1 / beta
        parser.add_argument('--weight', type = float, default = 10)
        opts,_ = parser.parse_known_args()
        parser.add_argument('--ge', type = float, default = 0.07)
        parser.add_argument('--u', type = float, default = 1.0)
    
    def return_parser(self):
        return self.parser