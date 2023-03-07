class args(object):
    def __init__(self, parser):
        self.parser = parser
        parser.add_argument('--gamma_a', type = float, default = 1, help = 'gamma_parameter_a')
        parser.add_argument('--gamma_b', type = float, default = 0.1, help = 'gamma_parameter_b')
        parser.add_argument('--n_hidden', type = int, default = 50)
        parser.add_argument('--test_size', type = float, default = 0.2, help = 'split ratio for dataset')

    def return_parser(self):
        return self.parser

'''
NOTE: settings from SVGD(NIPS 2016)
p(y | W, X, \gamma) = \prod_i^N  N(y_i | f(x_i; W), \gamma^{-1})
p(W | \lambda) = \prod_i N(w_i | 0, \lambda^{-1})
p(\gamma) = Gamma(\gamma | a0, b0)
p(\lambda) = Gamma(\lambda | a0, b0)
    
The posterior distribution is as follows:
p(W, \gamma, \lambda) = p(y | W, X, \gamma) p(W | \lambda) p(\gamma) p(\lambda) 
To avoid negative values of \gamma and \lambda, we update loggamma and loglambda instead.
'''