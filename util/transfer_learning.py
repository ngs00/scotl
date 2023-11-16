import json
from copy import deepcopy
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from util.dataset import *
from model.util import *
from model.gnn import get_embs


def load_exp_config(path_config_file):
    with open(path_config_file, 'r') as f:
        return json.load(f)


def calc_transferability(model_src, dataset_target, alpha=1.0):
    z = get_embs(model_src, dataset_target[:, :-1]).numpy()
    y = dataset_target[:, -1]
    model = Ridge(alpha=alpha)
    model.fit(z, y)
    train_loss = numpy.sqrt(mean_squared_error(y, model.predict(z)))
    l2_reg = numpy.sqrt(numpy.linalg.norm(model.coef_)**2)

    # return 1 / numpy.sqrt(mean_squared_error(y, model.predict(z)))
    return 1 / train_loss, l2_reg


def exec_tl(src_model, gnn, n_form_feats, dataset_target_train, tl_method, batch_size=32, init_lr=5e-4, l2_coeff=1e-6):
    _src_model = deepcopy(src_model)
    dim_emb = 64 if gnn == 'mpnn' else 128

    if tl_method == 'fit_reg_head':
        model = fit_reg_head(_src_model, dataset_target_train, n_form_feats, dim_emb,
                             batch_size=batch_size, n_epochs=300)
    elif tl_method == 'fine_tuning':
        model = fine_tuning(_src_model, dataset_target_train, n_form_feats, dim_emb,
                            batch_size=batch_size, init_lr=init_lr, n_epochs=300, l2_coeff=l2_coeff)
    else:
        raise KeyError

    return model
