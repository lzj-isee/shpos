class args(object):
    def __init__(self, parser):
        self.parser = parser
        parser.add_argument('--iters', type = int, default = 401)
        parser.add_argument('--particle_num', type = int, default = 1000)
        parser.add_argument('--batch_size', type = int, default = 1)
        # parser.add_argument('--init_std', type = float, default = 0.001)
        # kernel_h = 1 / ln(particle_num)
        parser.add_argument('--kernel_h', type = float, default = 0.2)
        parser.add_argument('--accHessBnd', type = float,default = 3e2)
        parser.add_argument('--accShrink', type = float, default = 0.2)
        parser.add_argument('--lr', type=float, default = 1e-3)
    
    def return_parser(self):
        return self.parser