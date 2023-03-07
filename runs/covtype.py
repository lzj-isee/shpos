import os
algo_list = []
#algo_list.append('SVGD')
#algo_list.append('SPOS')
#algo_list.append('SVRGPOS')
#algo_list.append('WNes')
#algo_list.append('HMCSPOS')
#algo_list.append('STORMSPOS')
#algo_list.append('ParaHMC')
#algo_list.append('ParaStormHMC')
run_all = True
lr_SVGD = [3e-6]
lr_SPOS, beta_SPOS = [6e-7], [5]
lr_SVRGPOS, beta_SVRGPOS = [8e-6], [5]
lr_WNes, accShrink_WNes, accHessBnd_WNes = [1.5e-6], [0.2], [3e4]
lr_HMCSPOS, weight_HMCSPOS, ge_HMCSPOS, u_HMCSPOS = [6e-5], [10], [0.02], [1]
lr_STORMSPOS, weight_STORMSPOS, ge_STORMSPOS, u_STORMSPOS, refresh_value_STORMSPOS, pow_STORMSPOS = \
    [7e-5], [10], [0.02], [1], 5000, 1
lr_ParaHMC, ge_ParaHMC, u_ParaHMC = [6e-5], [0.02], [1]
lr_ParaStormHMC, ge_ParaStormHMC, u_ParaStormHMC, refresh_value_ParaStormHMC, pow_ParaStormHMC = \
    [7e-5], [0.02], [1], 5000, 1
# covtype trian_num 581012
command = 'python3 main.py --gpu_ids 3 --save_folder results --task logistic_regression --dataset covtype \
    --eval_interval 10 --epochs 10 --particle_num 50 --batch_size 1000 --kernel_h 0.25 --seed 0 '
command_SVRG = 'python3 main.py --gpu_ids 3 --save_folder results --task logistic_regression --dataset covtype \
    --eval_interval 10 --epochs 5 --particle_num 50 --batch_size 1000 --kernel_h 0.25 --seed 0 '

for algo in algo_list:
    if 'SVGD' == algo:
        for lr in lr_SVGD:
            os.system(command + '--algorithm SVGD --lr {}'.format(lr))
    elif 'SPOS' == algo:
        for lr in lr_SPOS:
            for beta in beta_SPOS:
                os.system(command + '--algorithm SPOS --lr {} --beta {}'.format(lr, beta)) 
    elif 'SVRGPOS' == algo:
        for lr in lr_SVRGPOS:
            for beta in beta_SVRGPOS:
                os.system(command_SVRG + '--algorithm SVRGPOS --lr {} --beta {}'.format(lr, beta)) 
    elif 'WNes' == algo:
        for lr in lr_WNes:
            for accShrink in accShrink_WNes:
                for accHessBnd in accHessBnd_WNes:
                    os.system(command + '--algorithm WNes --lr {} --accShrink {} --accHessBnd {}'.format(lr, accShrink, accHessBnd))
    elif 'HMCSPOS' == algo:
        for lr in lr_HMCSPOS:
            for weight in weight_HMCSPOS:
                for ge in ge_HMCSPOS:
                    for u in u_HMCSPOS:
                        os.system(command + '--algorithm HMCSPOS --lr {} --weight {} --ge {} --u {}'.format(lr, weight, ge, u))
    elif 'STORMSPOS' == algo:
        for lr in lr_STORMSPOS:
            for weight in weight_STORMSPOS:
                for ge in ge_STORMSPOS:
                    for u in u_STORMSPOS:
                        os.system(command + '--algorithm STORMSPOS \
                        --refresh_value {} --pow {} \
                        --lr {} --weight {} --ge {} --u {}'.format(refresh_value_STORMSPOS,pow_STORMSPOS,lr, weight, ge, u))
    elif 'ParaHMC' == algo:
        for lr in lr_ParaHMC:
            for ge in ge_ParaHMC:
                for u in u_ParaHMC:
                    os.system(command + '--algorithm ParaHMC --lr {} --ge {} --u {}'.format(lr, ge, u))
    elif 'ParaStormHMC' == algo:
        for lr in lr_ParaStormHMC:
                for ge in ge_ParaStormHMC:
                    for u in u_ParaStormHMC:
                        os.system(command + '--algorithm ParaStormHMC \
                        --refresh_value {} --pow {} \
                        --lr {} --ge {} --u {}'.format(refresh_value_ParaStormHMC, pow_ParaStormHMC, lr, ge, u))

if run_all:
    for i in range(0, 20): 
        '''
        os.system('python3 main.py --algorithm SVGD --gpu_ids 2 --save_folder results_covtype \
            --task logistic_regression --dataset covtype --eval_interval 10 \
            --epochs 10 --particle_num 50 --batch_size 1000 --kernel_h 0.25 \
            --seed {} --lr 3e-6 '.format(i)) 

        os.system('python3 main.py --algorithm SPOS --gpu_ids 2 --save_folder results_covtype \
            --task logistic_regression --dataset covtype --eval_interval 10 \
            --epochs 10 --particle_num 50 --batch_size 1000 --kernel_h 0.25 \
            --seed {} --lr 6e-7 --beta 5'.format(i)) 

        os.system('python3 main.py --algorithm SVRGPOS --gpu_ids 2 --save_folder results_covtype \
            --task logistic_regression --dataset covtype --eval_interval 10 \
            --epochs 5 --particle_num 50 --batch_size 1000 --kernel_h 0.25 \
            --seed {} --lr 8e-6 --beta 5'.format(i)) 

        os.system('python3 main.py --algorithm WNes --gpu_ids 2 --save_folder results_covtype \
            --task logistic_regression --dataset covtype --eval_interval 10 \
            --epochs 10 --particle_num 50 --batch_size 1000 --kernel_h 0.25 \
            --seed {} --lr 1.5e-6 --accShrink 0.2 --accHessBnd 3e4'.format(i))

        os.system('python3 main.py --algorithm HMCSPOS --gpu_ids 2 --save_folder results_covtype \
            --task logistic_regression --dataset covtype --eval_interval 10 \
            --epochs 10 --particle_num 50 --batch_size 1000 --kernel_h 0.25 \
            --seed {} --lr 6e-5 --weight 10 --ge 0.02 --u 1'.format(i)) 

        os.system('python3 main.py --algorithm STORMSPOS --gpu_ids 2 --save_folder results_covtype \
            --task logistic_regression --dataset covtype --eval_interval 10 \
            --epochs 10 --particle_num 50 --batch_size 1000 --kernel_h 0.25 \
            --refresh_value 5000 --pow 1 \
            --seed {} --lr 7e-5 --weight 10 --ge 0.02 --u 1'.format(i)) 
        '''
        os.system('python3 main.py --algorithm ParaHMC --gpu_ids 2 --save_folder results_covtype \
            --task logistic_regression --dataset covtype --eval_interval 10 \
            --epochs 10 --particle_num 50 --batch_size 1000 --kernel_h 0.25 \
            --seed {} --lr 6e-5 --ge 0.02 --u 1'.format(i)) 

        os.system('python3 main.py --algorithm ParaStormHMC --gpu_ids 2 --save_folder results_covtype \
            --task logistic_regression --dataset covtype --eval_interval 10 \
            --epochs 10 --particle_num 50 --batch_size 1000 --kernel_h 0.25 \
            --refresh_value 5000 --pow 1 \
            --seed {} --lr 7e-5 --ge 0.02 --u 1'.format(i))