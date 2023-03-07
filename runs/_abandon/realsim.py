import os

lr_s = [1e-5]
for lr in lr_s:
    os.system('python3 main.py --algorithm SVGD --gpu_ids 3 --save_folder results \
        --task logistic_regression --dataset realsim --eval_interval 20 \
        --iters 10001 --particle_num 50 --batch_size 100 --kernel_h 0.25\
        --seed 0 --lr {}'.format(lr)) 

'''
lr_s = [6e-4]
beta_s = [10]
for lr in lr_s:
    for beta in beta_s:
        os.system('python3 main.py --algorithm SPOS --gpu_ids 3 --save_folder results \
            --task logistic_regression --dataset realsim --eval_interval 20 \
            --iters 10001 --particle_num 50 --batch_size 100 --kernel_h 0.25\
            --seed 0 --lr {} --beta {}'.format(lr, beta)) 
'''
'''
lr_s = [1.2e-3]
weight_s = [20]
ge_s = [0.01]
u_s = [0.1]
for lr in lr_s:
    for weight in weight_s:
        for ge in ge_s:
            for u in u_s:
                os.system('python3 main.py --algorithm SPOSHMC2 --gpu_ids 3 --save_folder results \
                    --task logistic_regression --dataset realsim --eval_interval 20 \
                    --iters 10001 --particle_num 50 --batch_size 100 --kernel_h 0.25\
                    --seed 0 --lr {} --weight {} --ge {} --u {}'.format(lr, weight, ge, u))
'''
'''
lr_s = [4e-3]
accShrink_s = [0.2]
accHessBnd_s = [3e2]
for lr in lr_s:
    for accShrink in accShrink_s:
        for accHessBnd in accHessBnd_s:
            os.system('python3 main.py --algorithm WNes --gpu_ids 3 --save_folder results \
                --task logistic_regression --dataset realsim --eval_interval 10 \
                --iters 10001 --particle_num 50 --batch_size 100 --kernel_h 0.25\
                --seed 0 --lr {} --accShrink {} --accHessBnd {}'.format(lr, accShrink, accHessBnd))
'''
'''
for i in range(0, 20): 
    os.system('python3 main.py --algorithm SVGD --gpu_ids 3 --save_folder results_realsim \
        --task logistic_regression --dataset realsim --eval_interval 10 \
        --iters 10001 --particle_num 50 --batch_size 100 --kernel_h 0.25 \
        --seed {} --lr 1e-3 '.format(i)) 
    os.system('python3 main.py --algorithm SPOS --gpu_ids 3 --save_folder results_realsim \
        --task logistic_regression --dataset realsim --eval_interval 10 \
        --iters 10001 --particle_num 50 --batch_size 100 --kernel_h 0.25 \
        --seed {} --lr 6e-4 --beta 10'.format(i)) 
    os.system('python3 main.py --algorithm SPOSHMC2 --gpu_ids 3 --save_folder results_realsim \
        --task logistic_regression --dataset realsim --eval_interval 10 \
        --iters 10001 --particle_num 50 --batch_size 100 --kernel_h 0.25 \
        --seed {} --lr 1.2e-3 --weight 20 --ge 0.01 --u 0.1'.format(i)) 
    os.system('python3 main.py --algorithm WNes --gpu_ids 3 --save_folder results_realsim \
        --task logistic_regression --dataset realsim --eval_interval 10 \
        --iters 10001 --particle_num 50 --batch_size 100 --kernel_h 0.25 \
        --seed {} --lr 4e-3 --accShrink 0.2 --accHessBnd 3e2'.format(i))
'''