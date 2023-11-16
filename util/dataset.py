import numpy
import pandas
import itertools
from tqdm import tqdm
from pymatgen.core import Structure
from util.chem import load_elem_attrs
from util.chem import get_fvec, get_crystal_graph


def load_dataset(path_metadata, path_structs, elem_attrs, idx_target, atomic_cutoff=4.0):
    metadata = pandas.read_excel(path_metadata).values.tolist()
    rbf_means = numpy.linspace(start=1.0, stop=atomic_cutoff, num=64)
    elem_attrs_fvec = load_elem_attrs()
    dataset = list()

    for i in tqdm(range(0, len(metadata))):
        struct = Structure.from_file('{}/{}.cif'.format(path_structs, metadata[i][0]))
        fvec = get_fvec(struct.composition.reduced_formula, elem_attrs_fvec)
        cg = get_crystal_graph(struct=struct,
                               elem_attrs=elem_attrs,
                               rbf_means=rbf_means,
                               idx=i,
                               fvec=fvec,
                               target=metadata[i][idx_target],
                               atomic_cutoff=atomic_cutoff)

        if cg is not None:
            dataset.append(cg)

    return dataset


def load_exp_dataset(path_dataset, elem_attrs,idx_target):
    dataset = pandas.read_excel(path_dataset).values.tolist()
    _dataset = list()

    for d in dataset:
        _dataset.append(numpy.hstack([get_fvec(d[0], elem_attrs), d[idx_target]]))

    return numpy.vstack(_dataset)


def get_dataset_info(dataset, idx_target):
    if dataset == 'mps':
        if idx_target == 2:
            return 'form_eng'
        elif idx_target == 3:
            return 'band_gap'
        elif idx_target == 4:
            return 'bulk_mod'
        elif idx_target == 5:
            return 'shear_mod'
        elif idx_target == 6:
            return 'poisson_ratio'
    elif dataset == 'mpl':
        if idx_target == 1:
            return 'form_eng'
        elif idx_target == 2:
            return 'band_gap'
    elif dataset == 'prb':
        if idx_target == 4:
            return 'g0w0'
    elif dataset == 'hoip':
        if idx_target == 1:
            return 'band_gap_gga'
        elif idx_target == 2:
            return 'band_gap_hse'
    elif dataset == 'nlhm':
        if idx_target == 1:
            return 'band_gap_gga'
        elif idx_target == 2:
            return 'band_gap_gllb'
    else:
        raise KeyError

    return None


def split_dataset(dataset, ratio_train=0.8, random_seed=None):
    if random_seed is not None:
        numpy.random.seed(random_seed)

    idx_rand = numpy.random.permutation(len(dataset))
    n_data_train = int(ratio_train * len(dataset))
    dataset_train = dataset[idx_rand[:n_data_train]]
    dataset_test = dataset[idx_rand[n_data_train:]]

    return dataset_train, dataset_test


def get_k_fold(dataset, n_folds, idx_fold, random_seed=None):
    if random_seed is not None:
        numpy.random.seed(random_seed)

    idx_rand = numpy.array_split(numpy.random.permutation(dataset.shape[0]), n_folds)
    idx_train = list(itertools.chain.from_iterable(idx_rand[:idx_fold] + idx_rand[idx_fold + 1:]))
    idx_test = idx_rand[idx_fold]

    return dataset[idx_train], dataset[idx_test]
