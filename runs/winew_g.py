import os

'''
lr_s = [0.8e-4]
for lr in lr_s:
    os.system('python3 main.py --algorithm SVGD --gpu_ids 3 --save_folder results \
        --task fnn2_regression_g --dataset WineWhite --eval_interval 10 \
        --iters 5001 --particle_num 20 --batch_size 64 --kernel_h 0.33 \
        --seed 0 --lr {}'.format(lr)) 
'''
'''
lr_s = [2e-5]
beta_s = [5]
for lr in lr_s:
    for beta in beta_s:
        os.system('python3 main.py --algorithm SPOS --gpu_ids 3 --save_folder results \
            --task fnn2_regression_g --dataset WineWhite --eval_interval 10 \
            --iters 5001 --particle_num 20 --batch_size 64 --kernel_h 0.33 \
            --seed 0 --lr {} --beta {}'.format(lr, beta)) 
'''
'''
lr_s = [2e-5]
accShrink_s = [0.2]
accHessBnd_s = [3e2]
for lr in lr_s:
    for accShrink in accShrink_s:
        for accHessBnd in accHessBnd_s:
            os.system('python3 main.py --algorithm WNes --gpu_ids 3 --save_folder results \
                --task fnn2_regression_g --dataset WineWhite --eval_interval 10 \
                --iters 5001 --particle_num 20 --batch_size 64 --kernel_h 0.33 \
                --seed 0 --lr {} --accShrink {} --accHessBnd {}'.format(lr, accShrink, accHessBnd))
'''
'''
lr_s = [1.8e-4]
weight_s = [10]
ge_s = [0.002]
u_s = [1]
for lr in lr_s:
    for weight in weight_s:
        for ge in ge_s:
            for u in u_s:
                os.system('python3 main.py --algorithm SPOSHMC2 --gpu_ids 3 --save_folder results \
                    --task fnn2_regression_g --dataset WineWhite --eval_interval 10 \
                    --iters 5001 --particle_num 20 --batch_size 64 --kernel_h 0.33 \
                    --seed 0 --lr {} --weight {} --ge {} --u {}'.format(lr, weight, ge, u))
'''

lr_s = [1.8e-4]
weight_s = [10]
ge_s = [0.002]
u_s = [1]
for lr in lr_s:
    for weight in weight_s:
        for ge in ge_s:
            for u in u_s:
                os.system('python3 main.py --algorithm SPOSHMC2_VR --gpu_ids 3 --save_folder results \
                    --task fnn2_regression_g --dataset WineWhite --eval_interval 10 \
                    --iters 5001 --particle_num 20 --batch_size 32 --kernel_h 0.33 \
                    --refresh_value 5000 --pow 0.5 \
                    --seed 0 --lr {} --weight {} --ge {} --u {}'.format(lr, weight, ge, u))

'''
for i in range(0, 20): 
    os.system('python3 main.py --algorithm SVGD --gpu_ids 1 --save_folder results_WineWhite \
        --task fnn2_regression_g --dataset WineWhite --eval_interval 10 \
        --iters 5001 --particle_num 20 --batch_size 100 --kernel_h 0.33 \
        --seed {} --lr 0.8e-4 '.format(i)) 
    os.system('python3 main.py --algorithm SPOS --gpu_ids 1 --save_folder results_WineWhite \
        --task fnn2_regression_g --dataset WineWhite --eval_interval 10 \
        --iters 5001 --particle_num 20 --batch_size 100 --kernel_h 0.33 \
        --seed {} --lr 2e-5 --beta 5'.format(i)) 
    os.system('python3 main.py --algorithm SPOSHMC2 --gpu_ids 1 --save_folder results_WineWhite \
        --task fnn2_regression_g --dataset WineWhite --eval_interval 10 \
        --iters 5001 --particle_num 20 --batch_size 100 --kernel_h 0.33 \
        --seed {} --lr 1.8e-4 --weight 10 --ge 0.002 --u 1'.format(i)) 
    os.system('python3 main.py --algorithm WNes --gpu_ids 1 --save_folder results_WineWhite \
        --task fnn2_regression_g --dataset WineWhite --eval_interval 10 \
        --iters 5001 --particle_num 20 --batch_size 100 --kernel_h 0.33 \
        --seed {} --lr 2e-5 --accShrink 0.2 --accHessBnd 3e2'.format(i))
'''