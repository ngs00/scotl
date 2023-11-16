from typing import Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.glob import global_mean_pool


class MEGNetConv(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        dim_edge: int,
        dim_state: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        share_weights: bool = False,
        **kwargs,
    ):
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim_edge = dim_edge
        self.dim_state = dim_state
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.share_weights = share_weights

        self.fc_node = Linear(self.in_channels + self.dim_edge + self.dim_state, self.out_channels)
        self.fc_edge = Linear(2 * self.in_channels + self.dim_edge + self.dim_state, self.dim_edge)
        self.fc_state = Linear(self.in_channels + self.dim_edge + self.dim_state, self.dim_state)

        self.reset_parameters()

    def reset_parameters(self):
        self.fc_node.reset_parameters()
        self.fc_edge.reset_parameters()
        self.fc_state.reset_parameters()

    def forward(self, x, u, edge_index, edge_attr, n_nodes, n_edges, batch):
        x_src = x[edge_index[0]]
        x_dest = x[edge_index[1]]
        u_x = torch.repeat_interleave(u, n_nodes.squeeze(1), dim=0)
        u_e = torch.repeat_interleave(u, n_edges.squeeze(1), dim=0)

        h_e = F.leaky_relu(self.fc_edge(torch.cat([x_src, x_dest, edge_attr, u_e], dim=1)))
        v_e = torch.repeat_interleave(global_mean_pool(h_e, batch[edge_index[0]]), n_nodes.squeeze(1), dim=0)
        h_x = F.leaky_relu(self.fc_node(torch.cat([v_e, x, u_x], dim=1)))

        u_e = global_mean_pool(h_e, batch[edge_index[0]])
        u_x = global_mean_pool(h_x, batch)
        h_u = F.leaky_relu(self.fc_state(torch.cat([u_e, u_x, u], dim=1)))

        return h_x, h_e, h_u

    def message(self, x_i, x_j, edge_attr):
        z = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return z

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
