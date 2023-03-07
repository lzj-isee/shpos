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
run_all = False
lr_SVGD = [1.5e-5]
lr_SPOS, beta_SPOS = [7e-6], [10]
lr_SVRGPOS, beta_SVRGPOS = [2.0e-5], [10]
lr_WNes, accShrink_WNes, accHessBnd_WNes = [1e-6], [0.2], [3e2]
lr_HMCSPOS, weight_HMCSPOS, ge_HMCSPOS, u_HMCSPOS = [1.6e-4], [5], [0.01], [1]
lr_STORMSPOS, weight_STORMSPOS, ge_STORMSPOS, u_STORMSPOS, refresh_value_STORMSPOS, pow_STORMSPOS = \
    [1.8e-4], [5], [0.01], [1], 5000, 0.3
lr_ParaHMC, ge_ParaHMC, u_ParaHMC = [1.6e-4], [0.01], [1]
lr_ParaStormHMC, ge_ParaStormHMC, u_ParaStormHMC, refresh_value_ParaStormHMC, pow_ParaStormHMC = \
    [1.8e-4], [0.01], [1], 5000, 0.3
# cadata trian_num 20640*0.8
command = 'python3 main.py --gpu_ids 3 --save_folder results --task fnn2_regression --dataset cadata \
    --eval_interval 10 --epochs 30 --particle_num 20 --batch_size 100 --kernel_h 0.33 --seed 0 '
command_SVRG = 'python3 main.py --gpu_ids 3 --save_folder results --task fnn2_regression --dataset cadata \
    --eval_interval 10 --epochs 15 --particle_num 20 --batch_size 100 --kernel_h 0.33 --seed 0 '

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
        os.system('python3 main.py --algorithm SVGD --gpu_ids 2 --save_folder results_cadata \
            --task fnn2_regression --dataset cadata --eval_interval 10 \
            --epochs 30 --particle_num 20 --batch_size 100 --kernel_h 0.33 \
            --seed {} --lr 1.5e-5 '.format(i)) 

        os.system('python3 main.py --algorithm SPOS --gpu_ids 2 --save_folder results_cadata \
            --task fnn2_regression --dataset cadata --eval_interval 10 \
            --epochs 30 --particle_num 20 --batch_size 100 --kernel_h 0.33 \
            --seed {} --lr 7e-6 --beta 10'.format(i)) 

        os.system('python3 main.py --algorithm SVRGPOS --gpu_ids 2 --save_folder results_cadata \
            --task fnn2_regression --dataset cadata --eval_interval 10 \
            --epochs 15 --particle_num 20 --batch_size 100 --kernel_h 0.33 \
            --seed {} --lr 2.0e-5 --beta 10'.format(i)) 

        os.system('python3 main.py --algorithm WNes --gpu_ids 2 --save_folder results_cadata \
            --task fnn2_regression --dataset cadata --eval_interval 10 \
            --epochs 30 --particle_num 20 --batch_size 100 --kernel_h 0.33 \
            --seed {} --lr 1.0e-6 --accShrink 0.2 --accHessBnd 3e2'.format(i))

        os.system('python3 main.py --algorithm HMCSPOS --gpu_ids 2 --save_folder results_cadata \
            --task fnn2_regression --dataset cadata --eval_interval 10 \
            --epochs 30 --particle_num 20 --batch_size 100 --kernel_h 0.33 \
            --seed {} --lr 1.6e-4 --weight 5 --ge 0.01 --u 1'.format(i)) 

        os.system('python3 main.py --algorithm STORMSPOS --gpu_ids 2 --save_folder results_cadata \
            --task fnn2_regression --dataset cadata --eval_interval 10 \
            --epochs 30 --particle_num 20 --batch_size 100 --kernel_h 0.33 \
            --refresh_value 5000 --pow 0.3 \
            --seed {} --lr 1.8e-4 --weight 5 --ge 0.01 --u 1'.format(i)) 
        '''
        os.system('python3 main.py --algorithm ParaHMC --gpu_ids 2 --save_folder results_cadata \
            --task fnn2_regression --dataset cadata --eval_interval 10 \
            --epochs 30 --particle_num 20 --batch_size 100 --kernel_h 0.33 \
            --seed {} --lr 1.6e-4 --ge 0.01 --u 1'.format(i)) 

        os.system('python3 main.py --algorithm ParaStormHMC --gpu_ids 2 --save_folder results_cadata \
            --task fnn2_regression --dataset cadata --eval_interval 10 \
            --epochs 30 --particle_num 20 --batch_size 100 --kernel_h 0.33 \
            --refresh_value 5000 --pow 0.3 \
            --seed {} --lr 1.8e-4 --ge 0.01 --u 1'.format(i))