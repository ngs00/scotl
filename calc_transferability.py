from util.transfer_learning import *


exp_config = load_exp_config('save/exp_config.json')
name_target_dataset = exp_config['name_target_dataset']
idx_target = exp_config['idx_target']
n_folds = exp_config['n_folds']


with open('save/tl_results_{}_{}.json'.format(name_target_dataset, idx_target), 'r') as f:
    exp_results = json.load(f)


dataset_target = load_exp_dataset(path_dataset='../../data/chem_data/{}.xlsx'.format(name_target_dataset),
                                  elem_attrs=load_elem_attrs(),
                                  idx_target=idx_target)


for k in range(0, n_folds):
    d_t_train, d_t_test = get_k_fold(dataset_target, n_folds, idx_fold=k, random_seed=exp_config['random_seed'])

    for src_dataset in exp_config['src_datasets']:
        for idx_src_target in exp_config['idx_src_targets'][src_dataset]:
            name_src_target = get_dataset_info(src_dataset, idx_src_target)
            for gnn in exp_config['gnn_models']:
                model_name = '{}_{}_{}'.format(gnn, src_dataset, name_src_target)
                print('Fold [{}/{}]\tSource model: {}'.format(k + 1, n_folds, model_name))

                path_src_model = 'save/model_src/{}.pt'.format(model_name)
                src_model = load_model(path_src_model, gnn, exp_config)
                transferability, l2_reg = calc_transferability(src_model, d_t_train)
                exp_results[model_name]['transferability'][k] = transferability
                exp_results[model_name]['l2_reg'][k] = l2_reg

with open('save/tl_results_{}_{}.json'.format(name_target_dataset, idx_target), 'w') as f:
    json.dump(exp_results, f)
