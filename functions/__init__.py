import os
import torch
import numpy as np

def str2bool(v):
    return v.lower() in ('true')

def create_dirs_if_not_exist(dir_list):
    if isinstance(dir_list, list):
        for dir in dir_list:
            if not os.path.exists(dir):
                os.makedirs(dir)
    else:
        if not os.path.exists(dir_list):
            os.makedirs(dir_list)

def save_settings(save_folder, settings):
    # save settings
    with open(save_folder+'/settings.md',mode='w') as f:
        for key in settings:
            f.write(key+': '+'{}'.format(settings[key])+' \n')

def save_final_results(save_folder, result_dict):
    # save results
    with open(os.path.join(save_folder, 'results.md'), mode='w') as f:
        for key in result_dict:
            f.write(key + ': ' + '{}'.format(result_dict[key])+ '\n')

def clear_log(save_folder):
    # clear log files
    if os.path.exists(save_folder):
        names = os.listdir(save_folder)
        for name in names:
            os.remove(save_folder+'/'+name)
        print('clear files in {}'.format(save_folder))
    else:
        pass

def gen_noise(eta, gamma, u, dim, device):
    # generate noise used in some HMC algorithms
    g_e = gamma * eta
    var1 = u * (1 - np.exp(-2 * g_e))
    var2 = (u * gamma**(-2)) * \
        (2*g_e + 4*np.exp(-g_e) - np.exp(-2*g_e) - 3)
    corr = (u * gamma**(-1)) * \
        (1 - 2*np.exp(-g_e) + np.exp(-2*g_e))
    xi = torch.randn(size = (1,dim), device = device)
    xi_v = xi * np.sqrt(var1)
    xi_x = corr/var1 * xi_v + np.sqrt(var2 - corr**2/var1) * torch.randn(size = (1,dim), device=device)
    return xi_x, xi_v  

@torch.no_grad()
def rbf_kernel_calc(h, x1, x2):
    # calculate RBF kernel
    result = torch.exp( - ((x1 - x2).norm(dim=1))**2/h)
    return result
    
@torch.no_grad()
def rbf_grads_x1_calc(h, x1, x2):
    # calculate $\nabla_{x1} RBF kernel$
    part_1 = rbf_kernel_calc(h,x1,x2)
    part_2 = -2*(x1-x2)/h
    result = part_1.view(-1,1) * part_2
    return result

def compute_ls_vector(len_param, sigma):    #NOTE: code from LSSGLD
    """
    Compute the vector of the first row of A^{-1/2}?
    A^{1/2}
    """
    # First row of the eigen matrix
    A_first_row = np.ones((len_param,), dtype='complex128')/np.sqrt(len_param)
    
    # First row of A
    c = np.zeros((len_param,), dtype='complex128')
    c[0] = 1.+2.*sigma
    c[1] = -1.*sigma
    c[-1] = -1.*sigma
    
    # Eigenvalues of A^{-1/2}
    eigvals = np.zeros((len_param,), dtype='float32')
    w1 = np.exp(1j*2*np.pi/len_param) # Unit root of 1
    for i in range(0, len_param):
        wi = w1**i
        eigvals[i] = np.sqrt(c[0]+c[1]*wi**(len_param-1) + c[-1]*wi**1)
    
    vec1 = A_first_row
    for i in range(len(vec1)):
        vec1[i] = vec1[i]*eigvals[i]
    
    # Compute first row of A^{-1/2}
    conv_vec = np.zeros((len_param,), dtype='float32')
    for i in range(0, len_param):
        # i-th column of A^T
        vec_tmp = np.zeros((len_param,), dtype='complex128')
        for ii in range(0, len_param):
            vec_tmp[ii] = w1**(i*ii)
        conv_vec[i] = np.real(np.dot(vec1, vec_tmp))/np.sqrt(len_param)
    
    return conv_vec
