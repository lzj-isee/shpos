class args(object):
    def __init__(self, parser):
        self.parser = parser
        parser.add_argument('--iters', type = int, default = 3001)
        parser.add_argument('--particle_num', type = int, default = 20)
        parser.add_argument('--batch_size', type = int, default = 32)
        parser.add_argument('--refresh_value', type = int, default = 3000)
        parser.add_argument('--pow', type = float, default = 1)
        parser.add_argument('--lr', type=float, default = 4e-4)
        opts,_ = parser.parse_known_args()
        parser.add_argument('--ge', type = float, default = 0.02)
        parser.add_argument('--u', type = float, default = 1.0)
        parser.add_argument('--kernel_h', type = float, default = 0.33)
    
    def return_parser(self):
        return self.parser