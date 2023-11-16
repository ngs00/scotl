import torch.utils.data
from torch.utils.data import TensorDataset
from model.gnn import *


def get_gnn(name_model, n_node_feats, n_edge_feats, n_form_feats):
    if name_model == 'gcn':
        return GCN(n_node_feats=n_node_feats,
                   n_form_feats=n_form_feats,
                   dim_out=1)
    elif name_model == 'gat':
        return GAT(n_node_feats=n_node_feats,
                   n_form_feats=n_form_feats,
                   dim_out=1)
    elif name_model == 'mpnn':
        return MPNN(n_node_feats=n_node_feats,
                    n_edge_feats=n_edge_feats,
                    n_form_feats=n_form_feats,
                    dim_out=1)
    elif name_model == 'cgcnn':
        return CGCNN(n_node_feats=n_node_feats,
                     n_edge_feats=n_edge_feats,
                     n_form_feats=n_form_feats,
                     dim_out=1)
    elif name_model == 'megnet':
        return MEGNet(n_node_feats=n_node_feats,
                      n_edge_feats=n_edge_feats,
                      n_form_feats=n_form_feats,
                      dim_out=1)
    else:
        raise KeyError


def fit_model(dataset, name_model, batch_size=64, init_lr=5e-4, n_epochs=300):
    model = get_gnn(name_model, dataset[0].x.shape[1], dataset[0].edge_attr.shape[1], dataset[0].fvec.shape[1]).cuda()
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=1e-6)
    criterion = torch.nn.MSELoss()

    for epoch in range(0, n_epochs):
        train_loss = fit(model, data_loader, optimizer, criterion)
        print('{}\tEpoch [{}/{}]\tTrain loss: {:.4f}'.format(name_model, epoch + 1, n_epochs, train_loss))

    return model


def load_model(path_model_file, gnn, exp_config):
    model = get_gnn(gnn, exp_config['n_node_feats'], exp_config['n_edge_feats'], exp_config['n_form_feats']).cuda()
    model.load_state_dict(torch.load(path_model_file))

    return model


def fit_reg_head(model, dataset, dim_fvec, dim_emb, batch_size=32, init_lr=5e-4, n_epochs=300):
    for param in model.parameters():
        param.requires_grad = False
    # model.fc_fvec1 = torch.nn.Linear(dim_fvec, 128)
    model.fc_fvec2 = torch.nn.Linear(128, dim_emb)
    model.fc2 = torch.nn.Linear(dim_emb, 32)
    model.fc3 = torch.nn.Linear(32, 1)
    model = model.cuda()

    # model.fc_fvec1.reset_parameters()
    model.fc_fvec2.reset_parameters()
    model.fc2.reset_parameters()
    model.fc3.reset_parameters()

    dataset_x = torch.tensor(dataset[:, :-1], dtype=torch.float)
    dataset_y = torch.tensor(dataset[:, -1], dtype=torch.float)
    data_loader = torch.utils.data.DataLoader(TensorDataset(dataset_x, dataset_y), batch_size=batch_size, shuffle=True)
    optimizer_head = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=1e-6)
    criterion_head = torch.nn.MSELoss()
    for epoch in range(0, n_epochs):
        fit_from_fvec(model, data_loader, optimizer_head, criterion_head)

    return model


def fine_tuning(model, dataset, n_form_feats, dim_emb, batch_size=32, init_lr=5e-4, n_epochs=300, l2_coeff=1e-6):
    # model.fc_fvec1 = torch.nn.Linear(n_form_feats, 128)
    # model.fc_fvec2 = torch.nn.Linear(128, dim_emb)
    # model.fc2 = torch.nn.Linear(dim_emb, 32)
    # model.fc3 = torch.nn.Linear(32, 1)
    model = model.cuda()

    model.fc_fvec1.reset_parameters()
    model.fc_fvec2.reset_parameters()
    model.fc2.reset_parameters()
    model.fc3.reset_parameters()

    dataset_x = torch.tensor(dataset[:, :-1], dtype=torch.float)
    dataset_y = torch.tensor(dataset[:, -1], dtype=torch.float)
    data_loader = torch.utils.data.DataLoader(TensorDataset(dataset_x, dataset_y), batch_size=batch_size, shuffle=True)
    optimizer_head = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=l2_coeff)
    criterion_head = torch.nn.L1Loss()
    for epoch in range(0, n_epochs):
        fit_from_fvec(model, data_loader, optimizer_head, criterion_head)

    return model
