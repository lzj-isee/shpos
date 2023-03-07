import os
SUFFIX = 'png'

# LR

os.system('python3 ./process_results/gen_line.py --main_folder results_20/results_a3a \
    --term test_nll --suffix {}'.format(SUFFIX))
os.system('python3 ./process_results/gen_line.py --main_folder results_20/results_a3a \
    --term test_error --suffix {}'.format(SUFFIX))

os.system('python3 ./process_results/gen_line.py --main_folder results_20/results_a8a \
    --term test_nll --suffix {}'.format(SUFFIX))
os.system('python3 ./process_results/gen_line.py --main_folder results_20/results_a8a \
    --term test_error --suffix {}'.format(SUFFIX))

os.system('python3 ./process_results/gen_line.py --main_folder results_20/results_covtype \
    --term test_nll --suffix {}'.format(SUFFIX))
os.system('python3 ./process_results/gen_line.py --main_folder results_20/results_covtype \
    --term test_error --suffix {}'.format(SUFFIX))

os.system('python3 ./process_results/gen_line.py --main_folder results_20/results_w8a \
    --term test_nll --suffix {}'.format(SUFFIX))
os.system('python3 ./process_results/gen_line.py --main_folder results_20/results_w8a \
    --term test_error --suffix {}'.format(SUFFIX))

os.system('python3 ./process_results/gen_line.py --main_folder results_20/results_ijcnn \
    --term test_nll --suffix {}'.format(SUFFIX))
os.system('python3 ./process_results/gen_line.py --main_folder results_20/results_ijcnn \
    --term test_error --suffix {}'.format(SUFFIX))

# NN
os.system('python3 ./process_results/gen_line.py --main_folder results_20/results_abalone \
    --term test_rmse --suffix {}'.format(SUFFIX))

os.system('python3 ./process_results/gen_line.py --main_folder results_20/results_cadata \
    --term test_rmse --suffix {}'.format(SUFFIX))

os.system('python3 ./process_results/gen_line.py --main_folder results_20/results_space \
    --term test_rmse --suffix {}'.format(SUFFIX))

os.system('python3 ./process_results/gen_line.py --main_folder results_20/results_superconduct \
    --term test_rmse --suffix {}'.format(SUFFIX))

os.system('python3 ./process_results/gen_line.py --main_folder results_20/results_energy \
    --term test_rmse --suffix {}'.format(SUFFIX))

os.system('python3 ./process_results/gen_line.py --main_folder results_20/results_WineWhite \
    --term test_rmse --suffix {}'.format(SUFFIX))

# Synthetic
'''
os.system('python3 ./process_results/gen_line.py --main_folder results_20/results_toy1 \
    --term W2 --suffix {}'.format(SUFFIX))
'''
'''
os.system('python3 ./process_results/gen_line.py --main_folder results_20/results_toydataset_3 \
    --term W2 --suffix {}'.format(SUFFIX))
'''