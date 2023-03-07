import os

'''
lr_s = [1.5e-6]
for lr in lr_s:
    os.system('python3 main.py --algorithm SVGD --gpu_ids 3 --save_folder results \
        --task fnn2_regression --dataset sgemm --eval_interval 20 \
        --iters 10001 --particle_num 20 --batch_size 1000 --kernel_h 0.33 \
        --seed 0 --lr {}'.format(lr)) 
'''
'''
lr_s = [2e-5]
beta_s = [4]
for lr in lr_s:
    for beta in beta_s:
        os.system('python3 main.py --algorithm SPOS --gpu_ids 3 --save_folder results \
            --task fnn2_regression --dataset sgemm --eval_interval 20 \
            --iters 10001 --particle_num 20 --batch_size 1000 --kernel_h 0.33 \
            --seed 0 --lr {} --beta {}'.format(lr, beta)) 
'''

lr_s = [5e-5]
weight_s = [5]
ge_s = [0.002]
u_s = [1]
for lr in lr_s:
    for weight in weight_s:
        for ge in ge_s:
            for u in u_s:
                os.system('python3 main.py --algorithm SPOSHMC2 --gpu_ids 3 --save_folder results \
                    --task fnn2_regression --dataset sgemm --eval_interval 20 \
                    --iters 10001 --particle_num 20 --batch_size 1000 --kernel_h 0.33 \
                    --seed 0 --lr {} --weight {} --ge {} --u {}'.format(lr, weight, ge, u))

'''
lr_s = [5e-7]
accShrink_s = [0.2]
accHessBnd_s = [3e2]
for lr in lr_s:
    for accShrink in accShrink_s:
        for accHessBnd in accHessBnd_s:
            os.system('python3 main.py --algorithm WNes --gpu_ids 3 --save_folder results \
                --task fnn2_regression --dataset sgemm --eval_interval 20 \
                --iters 10001 --particle_num 20 --batch_size 1000 --kernel_h 0.33 \
                --seed 0 --lr {} --accShrink {} --accHessBnd {}'.format(lr, accShrink, accHessBnd))
'''
'''
for i in range(0, 20): 
    os.system('python3 main.py --algorithm SVGD --gpu_ids 1 --save_folder results_sgemm \
        --task fnn2_regression --dataset sgemm --eval_interval 20 \
        --iters 10001 --particle_num 20 --batch_size 1000 --kernel_h 0.33 \
        --seed {} --lr 8e-5 '.format(i)) 
    os.system('python3 main.py --algorithm SPOS --gpu_ids 1 --save_folder results_sgemm \
        --task fnn2_regression --dataset sgemm --eval_interval 20 \
        --iters 10001 --particle_num 20 --batch_size 1000 --kernel_h 0.33 \
        --seed {} --lr 2e-5 --beta 4'.format(i)) 
    os.system('python3 main.py --algorithm SPOSHMC2 --gpu_ids 1 --save_folder results_sgemm \
        --task fnn2_regression --dataset sgemm --eval_interval 20 \
        --iters 10001 --particle_num 20 --batch_size 1000 --kernel_h 0.33 \
        --seed {} --lr 4e-4 --weight 5 --ge 0.02 --u 1'.format(i)) 
    os.system('python3 main.py --algorithm WNes --gpu_ids 1 --save_folder results_sgemm \
        --task fnn2_regression --dataset sgemm --eval_interval 20 \
        --iters 10001 --particle_num 20 --batch_size 1000 --kernel_h 0.33 \
        --seed {} --lr 2e-5 --accShrink 0.2 --accHessBnd 3e2'.format(i))
'''