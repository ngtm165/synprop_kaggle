import torch
import torch.nn as nn
from torch_geometric.nn.conv import GINConv
from torch_geometric.nn.pool import global_add_pool


class GNN(nn.Module):
    def __init__(
        self,
        node_in_feats,
        edge_in_feats,  # GIN không sử dụng edge features trực tiếp
        depth=5,
        node_hid_feats=300,
        readout_feats=1024,
        dr=0.1,
        readout_option=True,
    ):
        super(GNN, self).__init__()

        self.depth = depth

        self.project_node_feats = nn.Sequential(
            nn.Linear(node_in_feats, node_hid_feats), nn.ReLU()
        )

        self.gnn_layers = nn.ModuleList(
            [
                GINConv(
                    nn=torch.nn.Sequential(
                        nn.Linear(node_hid_feats, node_hid_feats),
                        nn.ReLU(),
                        nn.Linear(node_hid_feats, node_hid_feats),
                    )
                )
                for _ in range(self.depth)
            ]
        )

        self.sparsify = nn.Sequential(
            nn.Linear(node_hid_feats, readout_feats), nn.PReLU()
        )

        self.dropout = nn.Dropout(dr)
        self.readout_option = readout_option

    def forward(self, data):
        node_feats_orig = data.x
        batch = data.batch

        node_feats_init = self.project_node_feats(node_feats_orig)
        node_feats = node_feats_init

        for i in range(self.depth):
            node_feats = self.gnn_layers[i](node_feats, data.edge_index)

            if i < self.depth - 1:
                node_feats = nn.functional.relu(node_feats)

            node_feats = self.dropout(node_feats)

        readout = global_add_pool(node_feats, batch)

        if self.readout_option:
            readout = self.sparsify(readout)

        return readout