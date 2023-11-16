import time
from sklearn.metrics import r2_score
from util.chem import load_elem_attrs
from util.transfer_learning import *


# Experiment settings.
exp_config = load_exp_config('save/exp_config.json')
name_target_dataset = exp_config['name_target_dataset']
idx_target = exp_config['idx_target']
n_folds = exp_config['n_folds']


# Initialize a dictionary object to store the transfer learning results.
tl_results = dict()
for src_dataset in exp_config['src_datasets']:
    for idx_src_target in exp_config['idx_src_targets'][src_dataset]:
        name_src_target = get_dataset_info(dataset=src_dataset, idx_target=idx_src_target)
        for gnn in exp_config['gnn_models']:
            tl_results['{}_{}_{}'.format(gnn, src_dataset, name_src_target)] = {
                'transferability': list(),
                'rmse_train': list(),
                'rmse_test': list(),
                'r2_test': list(),
                'l2_reg': list()
            }


# Load the target experimental dataset.
dataset_target = load_exp_dataset(path_dataset='../../data/chem_data/{}.xlsx'.format(name_target_dataset),
                                  elem_attrs=load_elem_attrs(),
                                  idx_target=idx_target)


# Perform transfer learning for each source model on the k-fold leave-one-out cross-validation.
for k in range(0, n_folds):
    d_t_train, d_t_test = get_k_fold(dataset_target, n_folds, idx_fold=k, random_seed=exp_config['random_seed'])
    tl_vals = list()
    models = list()

    time_start = time.time()
    for src_dataset in exp_config['src_datasets']:
        for idx_src_target in exp_config['idx_src_targets'][src_dataset]:
            name_src_target = get_dataset_info(src_dataset, idx_src_target)
            for gnn in exp_config['gnn_models']:
                model_name = '{}_{}_{}'.format(gnn, src_dataset, name_src_target)
                # print('Fold [{}/{}]\tSource model: {}'.format(k + 1, n_folds, model_name))

                # Load source model.
                path_src_model = 'save/model_src/{}.pt'.format(model_name)
                src_model = load_model(path_src_model, gnn, exp_config)

                # Calculate transferability between the source model and the target experimental dataset.
                transferability, l2_reg = calc_transferability(src_model, d_t_train)

                tl_vals.append(transferability)
                models.append(model_name)

                # # Optimize model parameters based on transfer learning.
                # model = exec_tl(src_model, gnn, exp_config['n_form_feats'], d_t_train,
                #                 tl_method='fine_tuning', batch_size=64, init_lr=1e-3, l2_coeff=5e-6)
                # preds_train = predict_from_fvec(model, d_t_train[:, :-1]).numpy().flatten()
                # preds_test = predict_from_fvec(model, d_t_test[:, :-1]).numpy().flatten()
                #
                # # Evaluate the optimized model and store the evaluation metrics.
                # rmse_train = numpy.sqrt(mean_squared_error(d_t_train[:, -1], preds_train))
                # rmse_test = numpy.sqrt(mean_squared_error(d_t_test[:, -1], preds_test))
                # r2_test = r2_score(d_t_test[:, -1], preds_test)
                # tl_results[model_name]['transferability'].append(transferability)
                # tl_results[model_name]['rmse_train'].append(rmse_train)
                # tl_results[model_name]['rmse_test'].append(rmse_test)
                # tl_results[model_name]['r2_test'].append(r2_test)
                # tl_results[model_name]['l2_reg'].append(l2_reg)
                # print(r2_test)
    print(models[numpy.argmax(tl_vals)], numpy.max(tl_vals))

#     tl_results['exec_time'] = time.time() - time_start
#     print(tl_results['exec_time'])
#
# # Save the transfer learning results.
# with open('save/tl_results_{}_{}.json'.format(name_target_dataset, idx_target), 'w') as f:
#     json.dump(tl_results, f)
