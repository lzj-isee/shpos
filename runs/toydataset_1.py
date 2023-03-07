import os

'''
lr_s = [0.3]
for lr in lr_s:
    os.system('python3 main.py --algorithm SVGD --task toydataset_1 --gpu_ids 3 \
        --eval_interval 10 --save_folder results \
        --epochs 401 --particle_num 1000 --batch_size 1 --kernel_h 0.14 \
        --seed 0 --lr {}  '.format(lr)) 
'''
'''
lr_s = [0.02]
beta_s = [0.2]
for lr in lr_s:
    for beta in beta_s:
        os.system('python3 main.py --algorithm SPOS --task toydataset_1 --gpu_ids 3 \
            --eval_interval 10 --save_folder results\
            --epochs 401 --particle_num 1000 --batch_size 1 --kernel_h 0.14 \
            --seed 0 --lr {} --beta {}'.format(lr, beta)) 
'''

'''
lr_s = [0.06]
weight_s = [10]
ge_s = [0.07]
for lr in lr_s:
    for weight in weight_s:
        for ge in ge_s:
            os.system('python3 main.py --algorithm SPOSHMC2 --task toydataset_1 --gpu_ids 3 \
                --eval_interval 10 --save_folder results \
                --epochs 401 --particle_num 1000 --batch_size 1 --kernel_h 0.14 \
                --seed 0 --lr {} --weight {} --ge {}'.format(lr, weight, ge)) 
'''
'''
lr_s = [0.06]
ge_s = [0.07]
for lr in lr_s:
    for ge in ge_s:
        os.system('python3 main.py --algorithm ParaHMC --task toydataset_1 --gpu_ids 3 \
            --eval_interval 10 --save_folder results \
            --epochs 401 --particle_num 1000 --batch_size 1 --kernel_h 0.14 \
            --seed 0 --lr {} --ge {}'.format(lr, ge)) 
'''
'''
lr_s = [0.6]
accShrink_s = [0.2]
accHessBnd_s = [3e1]
for lr in lr_s:
    for accShrink in accShrink_s:
        for accHessBnd in accHessBnd_s:
            os.system('python3 main.py --algorithm WNes --task toydataset_1 --gpu_ids 3 \
                --eval_interval 10 --save_folder results \
                --epochs 401 --particle_num 1000 --batch_size 1 --kernel_h 0.14 \
                --seed 0 --lr {} --accShrink {} --accHessBnd {}'.format(lr, accShrink, accHessBnd))
'''

for i in range(0, 20): 
    os.system('python3 main.py --algorithm SVGD --task toydataset_1 --gpu_ids 3 \
        --eval_interval 10 --save_folder ./results_toy1\
        --particle_num 1000 --batch_size 1 \
        --seed {} --lr 0.2 --epochs 401 --kernel_h 0.14'.format(i))  

    os.system('python3 main.py --algorithm SPOS --task toydataset_1 --gpu_ids 3 \
        --eval_interval 10 --save_folder results\
        --particle_num 1000 --batch_size 1 \
        --seed {} --lr 0.02 --epochs 401 --kernel_h 0.14 --beta 0.2'.format(i)) 
    
    os.system('python3 main.py --algorithm WNes --task toydataset_1 --gpu_ids 3 \
        --eval_interval 10 --save_folder ./results_toy1 \
        --epochs 401 --particle_num 1000 --batch_size 1 --kernel_h 0.14 \
        --seed {} --lr 0.6 --accShrink 0.2 --accHessBnd 3e1'.format(i))

    os.system('python3 main.py --algorithm HMCSPOS --task toydataset_1 --gpu_ids 3 \
        --eval_interval 10 --save_folder ./results_toy1 \
        --particle_num 1000 --batch_size 1 \
        --seed {} --lr 0.06 --epochs 401 --kernel_h 0.14 --weight 10 --ge 0.07'.format(i)) 

    os.system('python3 main.py --algorithm ParaHMC --task toydataset_1 --gpu_ids 3 \
        --eval_interval 10 --save_folder ./results_toy1 \
        --particle_num 1000 --batch_size 1 \
        --seed {} --lr 0.06 --epochs 401 --kernel_h 0.14 --ge 0.07'.format(i)) 

