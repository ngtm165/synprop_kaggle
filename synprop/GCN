import torch
import torch.nn as nn
from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn.pool import global_add_pool

class GNN(nn.Module):
    def __init__(
        self,
        node_in_feats,
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
                GCNConv(node_hid_feats, node_hid_feats)  # Output channels added!
                for _ in range(self.depth)
            ]
        )

        self.sparsify = nn.Sequential(
            nn.Linear(node_hid_feats, readout_feats), nn.PReLU()
        )

        self.dropout = nn.Dropout(dr)
        self.readout_option = readout_option

    # def forward(self, data):
    #     node_feats_orig = data.x
    #     batch = data.batch

    #     node_feats_init = self.project_node_feats(node_feats_orig)
    #     node_feats = node_feats_init

    #     for i in range(self.depth):
    #         node_feats = self.gnn_layers[i](node_feats, data.edge_index)

    #         if i < self.depth - 1:
    #             node_feats = nn.functional.relu(node_feats)

    #         node_feats = self.dropout(node_feats)

    #     readout = global_add_pool(node_feats, batch)

    #     if self.readout_option:
    #         readout = self.sparsify(readout)

    #     return readout
    
    def forward(self, data):
        node_feats_orig = data.x
        # edge_feats_orig = data.edge_attr  # No edge features for basic GCN
        batch = data.batch
        edge_index = data.edge_index # Essential for GCN

        node_feats_init = self.project_node_feats(node_feats_orig)
        node_feats = node_feats_init
        # edge_feats = self.project_edge_feats(edge_feats_orig) # No edge features


        for i in range(self.depth):
            node_feats = self.gnn_layers[i](node_feats, edge_index) # Pass edge_index

            if i < self.depth - 1:
                node_feats = nn.functional.relu(node_feats)

            node_feats = self.dropout(node_feats)

        readout = global_add_pool(node_feats, batch)

        if self.readout_option:
            readout = self.sparsify(readout)

        return readout
