## Full Scale Experiments -- This script was run in the creation of the final version of the paper
query_type = ['var_ratio', 'entropy', 'margin_score', 'random']

### Set 1
datasets = [ 'ag_news', 'yahoo_answers',  'yelp_review_full', 'amazon_review_full']
lr = [0.25, 0.1, 0.1, 0.1]
num_epochs = [10, 10, 10, 10]
init_train_percent = [0.04, 0.02, 0.01, 0.005]
acq_iters = [4, 9, 19, 39]
dit_arr = [0.01, 0.005, 0.0025, 0.00125]
cnt = 0

for run in range(3):
    m = 'FastText'
    for it, acq, dit in zip(init_train_percent, acq_iters, dit_arr):
        for seed in range(3):
            for dset, l, ne in zip(datasets,lr,num_epochs):
                for q in query_type:
                    print(f'echo "python main.py --model {m} --dataset {dset} --lr {l} --num_epochs {ne} --query_type {q} --init_train_percent {it} --num_acquise_percent {it} --query_iter {acq} --seed {seed} --exp_name model+{m}+dataset+{dset}+query_type+{q}+lr+{l}+epochs+{ne}+itp_nacq+{it}+itr+{acq}+seed+{seed}+run+{run}+" >> {dset}_{m}_run{run}_exp.sh')
                    print(f'echo "python main.py --model {m} --dataset {dset} --lr {l} --num_epochs {ne} --query_type {q} --init_train_percent {it} --num_acquise_percent {it} --query_iter {acq} --seed {seed} --num_ensemble 5 --exp_name model+{m}+dataset+{dset}+query_type+{q}+lr+{l}+epochs+{ne}+itp_nacq+{it}+itr+{acq}+num_ensemble+5+seed+{seed}+run+{run}+" >> {dset}_{m}_run{run}_exp.sh')
                    print(f'echo "python main.py --model {m} --dataset {dset} --lr {l} --num_epochs {ne} --query_type {q} --init_train_percent {it} --num_acquise_percent {it} --num_delete_percent {dit} --query_iter {acq} --seed {seed} --exp_name model+{m}+dataset+{dset}+query_type+{q}+lr+{l}+epochs+{ne}+itp_nacq+{it}+dit+{dit}+itr+{acq}+seed+{seed}+run+{run}+" >> {dset}_{m}_run{run}_exp.sh')

m = 'NaiveBayes'
for it, acq, dit in zip(init_train_percent, acq_iters, dit_arr):
    for seed in range(3):
        for dset in datasets:
            for q in query_type:
                print(f'echo "python main.py --model {m} --dataset {dset} --query_type {q} --init_train_percent {it} --num_acquise_percent {it} --query_iter {acq} --seed {seed} --exp_name model+{m}+dataset+{dset}+query_type+{q}+itp_nacq+{it}+itr+{acq}+seed+{seed}+run+0+" >> {dset}_{m}_run0_exp.sh')
                print(f'echo "python main.py --model {m} --dataset {dset} --query_type {q} --init_train_percent {it} --num_acquise_percent {it} --num_delete_percent {dit} --query_iter {acq} --seed {seed} --exp_name model+{m}+dataset+{dset}+query_type+{q}+itp_nacq+{it}+dit+{dit}+itr+{acq}+seed+{seed}+run+0+" >> {dset}_{m}_run0_exp.sh')

### Set 2
datasets = ['dbpedia', 'sogou_news', 'yelp_review_polarity', 'amazon_review_polarity']
lr = [0.5, 0.25, 0.1, 0.1]
num_epochs = [10, 10, 10, 10]
init_train_percent = [0.02, 0.01, 0.005, 0.0025]
acq_iters = [4, 9, 19, 39]
dit_arr = [0.005, 0.0025, 0.00125, 0.000625]

for run in range(3):
    m = 'FastText'
    for it, acq, dit in zip(init_train_percent, acq_iters, dit_arr):
        for seed in range(3):
            for dset, l, ne in zip(datasets,lr,num_epochs):
                for q in query_type:
                    print(f'echo "python main.py --model {m} --dataset {dset} --lr {l} --num_epochs {ne} --query_type {q} --init_train_percent {it} --num_acquise_percent {it} --query_iter {acq} --seed {seed} --exp_name model+{m}+dataset+{dset}+query_type+{q}+lr+{l}+epochs+{ne}+itp_nacq+{it}+itr+{acq}+seed+{seed}+run+{run}+" >> {dset}_{m}_run{run}_exp.sh')
                    print(f'echo "python main.py --model {m} --dataset {dset} --lr {l} --num_epochs {ne} --query_type {q} --init_train_percent {it} --num_acquise_percent {it} --query_iter {acq} --seed {seed} --num_ensemble 5 --exp_name model+{m}+dataset+{dset}+query_type+{q}+lr+{l}+epochs+{ne}+itp_nacq+{it}+itr+{acq}+num_ensemble+5+seed+{seed}+run+{run}+" >> {dset}_{m}_run{run}_exp.sh')
                    print(f'echo "python main.py --model {m} --dataset {dset} --lr {l} --num_epochs {ne} --query_type {q} --init_train_percent {it} --num_acquise_percent {it} --num_delete_percent {dit} --query_iter {acq} --seed {seed} --exp_name model+{m}+dataset+{dset}+query_type+{q}+lr+{l}+epochs+{ne}+itp_nacq+{it}+dit+{dit}+itr+{acq}+seed+{seed}+run+{run}+" >> {dset}_{m}_run{run}_exp.sh')
m = 'NaiveBayes'
for it, acq, dit in zip(init_train_percent, acq_iters, dit_arr):
    for seed in range(3):
        for dset in datasets:
            for q in query_type:
                print(f'echo "python main.py --model {m} --dataset {dset} --query_type {q} --init_train_percent {it} --num_acquise_percent {it} --query_iter {acq} --seed {seed} --exp_name model+{m}+dataset+{dset}+query_type+{q}+itp_nacq+{it}+itr+{acq}+seed+{seed}+run+0+" >> {dset}_{m}_run0_exp.sh')
                print(f'echo "python main.py --model {m} --dataset {dset} --query_type {q} --init_train_percent {it} --num_acquise_percent {it} --num_delete_percent {dit} --query_iter {acq} --seed {seed} --exp_name model+{m}+dataset+{dset}+query_type+{q}+itp_nacq+{it}+dit+{dit}+itr+{acq}+seed+{seed}+run+0+" >> {dset}_{m}_run0_exp.sh')
