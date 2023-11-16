import torch
import torch.utils.data
from torch.nn.functional import leaky_relu, normalize
from torch_geometric.loader import DataLoader
from torch_geometric.nn.conv import *
from torch_geometric.nn.glob import global_mean_pool


def fit(model, data_loader, optimizer, criterion):
    train_loss = 0

    model.train()
    for batch in data_loader:
        batch = batch.cuda()
        preds, hg, hf = model(batch, train=True)

        loss = criterion(batch.y, preds)
        loss_dist = (hf - hg)**2
        loss_dist[loss_dist < 0.01] = 0
        loss += 1e-1 * torch.mean(loss_dist)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.detach().item()

    return train_loss / len(data_loader)


def fit_from_fvec(model, data_loader, optimizer, criterion):
    train_loss = 0

    model.train()
    for x, y in data_loader:
        preds = model.forward_from_fvec(x.cuda())
        loss = criterion(y.view(-1, 1).cuda(), preds)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.detach().item()

    return train_loss / len(data_loader)


def predict(model, dataset):
    data_loader = DataLoader(dataset, batch_size=128)

    model.eval()
    with torch.no_grad():
        return torch.vstack([model(batch.cuda()) for batch in data_loader]).cpu()


def predict_from_fvec(model, x):
    model.eval()
    with torch.no_grad():
        return model.forward_from_fvec(torch.tensor(x, dtype=torch.float).cuda()).cpu()


def get_embs(model, x):
    model.eval()
    with torch.no_grad():
        return model.forward_fvec(torch.tensor(x, dtype=torch.float).cuda()).cpu()


class GNN(torch.nn.Module):
    def __init__(self, n_form_feats):
        super(GNN, self).__init__()
        self.fc_fvec1 = torch.nn.Linear(n_form_feats, 128)
        self.fc_fvec2 = torch.nn.Linear(128, 128)

    def emb(self, g):
        hx = self.fc1(g.x)
        hx = leaky_relu(self.gc1(hx, g.edge_index))
        hx = leaky_relu(self.gc2(hx, g.edge_index))
        h = normalize(global_mean_pool(hx, g.batch), p=2, dim=1)

        return h

    def forward_fvec(self, fvec):
        h = leaky_relu(self.fc_fvec1(fvec))
        out = normalize(self.fc_fvec2(h), p=2, dim=1)

        return out

    def forward(self, g, train=False):
        hg = self.emb(g)
        h = leaky_relu(self.fc2(hg))
        out = self.fc3(h)

        if train:
            return out, hg, self.forward_fvec(g.fvec)
        else:
            return out

    def forward_from_fvec(self, fvec):
        h = leaky_relu(self.fc_fvec1(fvec))
        h = normalize(self.fc_fvec2(h), p=2, dim=1)
        h = leaky_relu(self.fc2(h))
        out = self.fc3(h)

        return out


class EdgeGNN(GNN):
    def emb(self, g):
        hx = self.fc1(g.x)
        hx = leaky_relu(self.gc1(hx, g.edge_index, g.edge_attr))
        hx = leaky_relu(self.gc2(hx, g.edge_index, g.edge_attr))
        h = normalize(global_mean_pool(hx, g.batch), p=2, dim=1)

        return h


class GCN(GNN):
    def __init__(self, n_node_feats, n_form_feats, dim_out):
        super(GCN, self).__init__(n_form_feats)
        self.fc1 = torch.nn.Linear(n_node_feats, 128)
        self.gc1 = GCNConv(128, 128)
        self.gc2 = GCNConv(128, 128)
        self.fc2 = torch.nn.Linear(128, 32)
        self.fc3 = torch.nn.Linear(32, dim_out)


class GAT(GNN):
    def __init__(self, n_node_feats, n_form_feats, dim_out):
        super(GAT, self).__init__(n_form_feats)
        self.fc1 = torch.nn.Linear(n_node_feats, 128)
        self.gc1 = GATv2Conv(128, 128)
        self.gc2 = GATv2Conv(128, 128)
        self.fc2 = torch.nn.Linear(128, 32)
        self.fc3 = torch.nn.Linear(32, dim_out)


class MPNN(EdgeGNN):
    def __init__(self, n_node_feats, n_edge_feats, n_form_feats, dim_out):
        super(MPNN, self).__init__(n_form_feats)
        self.fc_fvec = torch.nn.Linear(n_form_feats, 64)
        self.fc1 = torch.nn.Linear(n_node_feats, 64)
        self.efc1 = torch.nn.Sequential(torch.nn.Linear(n_edge_feats, 64),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(64, 64 * 64))
        self.gc1 = NNConv(64, 64, self.efc1)
        self.efc2 = torch.nn.Sequential(torch.nn.Linear(n_edge_feats, 64),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(64, 64 * 64))
        self.gc2 = NNConv(64, 64, self.efc2)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, dim_out)


class CGCNN(EdgeGNN):
    def __init__(self, n_node_feats, n_edge_feats, n_form_feats, dim_out):
        super(CGCNN, self).__init__(n_form_feats)
        self.fc1 = torch.nn.Linear(n_node_feats, 128)
        self.gc1 = CGConv(128, n_edge_feats)
        self.gc2 = CGConv(128, n_edge_feats)
        self.fc2 = torch.nn.Linear(128, 32)
        self.fc3 = torch.nn.Linear(32, dim_out)


class MEGNet(EdgeGNN):
    def __init__(self, n_node_feats, n_edge_feats, n_form_feats, dim_out):
        super(MEGNet, self).__init__(n_form_feats)
        self.fc1 = torch.nn.Linear(n_node_feats, 128)
        self.gc1 = CGConv(128, n_edge_feats)
        self.gc2 = CGConv(128, n_edge_feats)
        self.fc2 = torch.nn.Linear(128, 32)
        self.fc3 = torch.nn.Linear(32, dim_out)
