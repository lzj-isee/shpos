import os

'''
lr_s = [40]
for lr in lr_s:
    os.system('python3 main.py --algorithm SVGD --task toydataset_3 --gpu_ids 3 \
        --eval_interval 5 --save_folder results \
        --iters 401 --particle_num 1000 --batch_size 1 --kernel_h 0.14 \
        --seed 0 --lr {}  '.format(lr)) 
'''
'''
lr_s = [3,4,5]
beta_s = [15]
for lr in lr_s:
    for beta in beta_s:
        os.system('python3 main.py --algorithm SPOS --task toydataset_3 --gpu_ids 3 \
            --eval_interval 5 --save_folder results\
            --iters 401 --particle_num 1000 --batch_size 1 --kernel_h 0.14 \
            --seed 0 --lr {} --beta {}'.format(lr, beta)) 
'''
'''
lr_s = [0.47]
weight_s = [20]
ge_s = [0.6]
u_s = [0.1]
for lr in lr_s:
    for weight in weight_s:
        for ge in ge_s:
            for u in u_s:
                os.system('python3 main.py --algorithm SPOSHMC2 --task toydataset_3 --gpu_ids 3 \
                    --eval_interval 5 --save_folder results \
                    --iters 401 --particle_num 1000 --batch_size 1 --kernel_h 0.14 \
                    --seed 1 --lr {} --weight {} --ge {} --u {}'.format(lr, weight, ge, u)) 
'''
'''
lr_s = [20]
accShrink_s = [0.2]
accHessBnd_s = [30]
for lr in lr_s:
    for accShrink in accShrink_s:
        for accHessBnd in accHessBnd_s:
            os.system('python3 main.py --algorithm WNes --task toydataset_3 --gpu_ids 3 \
                --eval_interval 5 --save_folder results \
                --iters 401 --particle_num 1000 --batch_size 1 --kernel_h 0.14 \
                --seed 0 --lr {} --accShrink {} --accHessBnd {}'.format(lr, accShrink, accHessBnd))
'''
  
for i in range(10, 20): 
    #os.system('python3 main.py --algorithm SVGD --task toydataset_3 --gpu_ids 3 \
    #    --eval_interval 5 --save_folder results1\
    #    --particle_num 1000 --batch_size 1 \
    #    --seed {} --lr 0.3 --iters 401 --kernel_h 0.14'.format(i))  
    #os.system('python3 main.py --algorithm SPOS  --gpu_ids 3 --save_folder results_toydataset_3 \
    #    --task toydataset_3 --eval_interval 10 \
    #    --iters 401 --particle_num 1000 --batch_size 1 --kernel_h 0.14 \
    #    --seed {} --lr 3 --beta 15'.format(i)) 
    os.system('python3 main.py --algorithm SPOSHMC2 --gpu_ids 3 --save_folder results_toydataset_3_d \
        --task toydataset_3 --eval_interval 10 \
        --iters 401 --particle_num 1000 --batch_size 1 --kernel_h 0.14 \
        --seed {} --lr 0.3 --weight 0 --ge 0.6 --u 1'.format(i))
    os.system('python3 main.py --algorithm SPOSHMC2 --gpu_ids 3 --save_folder results_toydataset_3_d \
        --task toydataset_3 --eval_interval 10 \
        --iters 401 --particle_num 1000 --batch_size 1 --kernel_h 0.14 \
        --seed {} --lr 0.8 --weight 1 --ge 0.6 --u 0.1'.format(i))
    os.system('python3 main.py --algorithm SPOSHMC2 --gpu_ids 3 --save_folder results_toydataset_3_d \
        --task toydataset_3 --eval_interval 10 \
        --iters 401 --particle_num 1000 --batch_size 1 --kernel_h 0.14 \
        --seed {} --lr 0.8 --weight 2 --ge 0.6 --u 0.1'.format(i))
    os.system('python3 main.py --algorithm SPOSHMC2 --gpu_ids 3 --save_folder results_toydataset_3_d \
        --task toydataset_3 --eval_interval 10 \
        --iters 401 --particle_num 1000 --batch_size 1 --kernel_h 0.14 \
        --seed {} --lr 0.8 --weight 5 --ge 0.6 --u 0.1'.format(i))
    os.system('python3 main.py --algorithm SPOSHMC2 --gpu_ids 3 --save_folder results_toydataset_3_d \
        --task toydataset_3 --eval_interval 10 \
        --iters 401 --particle_num 1000 --batch_size 1 --kernel_h 0.14 \
        --seed {} --lr 0.45 --weight 20 --ge 0.6 --u 0.1'.format(i))
    #os.system('python3 main.py --algorithm WNes --task toydataset_3 --gpu_ids 2 \
    #    --eval_interval 5 --save_folder results_20/results_toy1 \
    #    --iters 401 --particle_num 1000 --batch_size 1 --kernel_h 0.14 \
    #    --seed {} --lr 0.4 --accShrink 0.2 --accHessBnd 3e2'.format(i)) 
