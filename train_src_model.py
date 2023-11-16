import torch
from util.chem import load_elem_attrs
from util.dataset import *
from model.util import fit_model


# Experiment settings.
name_datasets = ['prb', 'hoip', 'mps', 'mpl']
idx_targets = {'prb': [4], 'hoip': [2], 'mps': [2, 3, 4, 5], 'mpl': [1, 2]}


models = ['gat', 'cgcnn', 'megnet']
elem_attrs = load_elem_attrs('res/matscholar-embedding.json')
path_src_model = 'save/model_src/{}_{}_{}.pt'
n_epochs = 500

for name_dataset in name_datasets:
    for idx_target in idx_targets[name_dataset]:
        name_target = get_dataset_info(name_dataset, idx_target)
        results = {model: dict() for model in models}

        # Load a source dataset.
        dataset_src = load_dataset(path_metadata='../../data/chem_data/{}/metadata.xlsx'.format(name_dataset),
                                   path_structs='../../data/chem_data/{}'.format(name_dataset),
                                   elem_attrs=elem_attrs,
                                   idx_target=idx_target)
        torch.save(dataset_src, 'save/dataset/{}_{}.pt'.format(name_dataset, name_target))
        dataset_src = torch.load('save/dataset/{}_{}.pt'.format(name_dataset, name_target))

        # Train source models.
        print(name_dataset, idx_target, name_target)
        for name_model in models:
            print('--------------------- {} {} {} ---------------------'.format(name_dataset, idx_target, name_model))
            model_src = fit_model(dataset_src, name_model, n_epochs=n_epochs)
            torch.save(model_src.state_dict(), path_src_model.format(name_model, name_dataset, name_target))
