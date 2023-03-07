class args(object):
    def __init__(self, parser):
        self.parser = parser
        #parser.add_argument('--weight_decay', type = float, default = 1)
        parser.add_argument('--n_hidden', type = int, default = 50)
        parser.add_argument('--test_size', type = float, default = 0.2, help = 'split ratio for dataset')

    def return_parser(self):
        return self.parser