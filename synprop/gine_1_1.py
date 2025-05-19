import torch
import torch.nn as nn
import torch.nn.functional as F # Thêm F cho ReLU giữa các lớp
from torch_geometric.nn.pool import global_add_pool
from torch_geometric.nn import MessagePassing # Import lớp cơ sở

# --- Định nghĩa lớp GNN lai: DMPNN Message & GINE Update ---
class DMPNNInspiredGINEConv(MessagePassing):
    """
    Lớp GNN lấy cảm hứng từ D-MPNN (bước tạo message) và GINE (cấu trúc cập nhật nút).
    Message m_{j->i} = message_nn(cat(x_j, edge_attr))     (Bước 1 - D-MPNN style)
    Aggregate: sum                                          (Bước 2 - GINE style)
    Update x'_i = nn((1+eps)x_i + sum(m_{j->i}))            (Bước 3 - GINE style)
    Activation for nn and message_nn is internal.
    """
    def __init__(self, nn: torch.nn.Module, message_nn: torch.nn.Module,
                 eps: float = 0., train_eps: bool = False,
                 aggr: str = 'add', **kwargs): # 'add' là sum trong PyG
        super().__init__(aggr=aggr, **kwargs)
        self.nn = nn # MLP h_theta cuối cùng, giống GINEConv
        self.message_nn = message_nn # Mạng nơ-ron để tạo message
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        def _reset_module_params(module_seq):
            for layer in module_seq:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        _reset_module_params(self.nn)
        _reset_module_params(self.message_nn)
        if hasattr(self.eps, 'data'):
            self.eps.data.fill_(self.initial_eps)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        # Bước 1 (message) & 2 (aggregate)
        aggregated_messages = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        # Bước 3 (update) kiểu GINE
        out = (1 + self.eps) * x + aggregated_messages
        out = self.nn(out) # MLP cuối (nn chứa activation bên trong)
        return out

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        # Bước 1: Tạo thông điệp kiểu D-MPNN (cat)
        input_message = torch.cat([x_j, edge_attr], dim=-1)
        # input_message = ([edge_attr])

        # message_nn chứa activation bên trong
        return self.message_nn(input_message)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn}, message_nn={self.message_nn})'

# --- Lớp GNN tổng thể ĐÃ ĐƯỢC MODIFY ---
class GNN(nn.Module):
    def __init__(
        self,
        node_in_feats,
        edge_in_feats,
        depth=5,
        node_hid_feats=300,
        edge_hid_feats=300, # <<< THÊM THAM SỐ NÀY
        readout_feats=1024,
        dr=0.1,
        readout_option=True,
        train_eps: bool = False # <<< THÊM THAM SỐ NÀY
    ):
        super(GNN, self).__init__()

        self.depth = depth
        self.readout_option = readout_option

        self.project_node_feats = nn.Sequential(
            nn.Linear(node_in_feats, node_hid_feats), nn.ReLU() # Có ReLU (Activation)
        )

        # project_edge_feats sẽ chiếu edge_in_feats sang edge_hid_feats
        self.project_edge_feats = nn.Sequential(
            nn.Linear(edge_in_feats, edge_hid_feats) # Output là edge_hid_feats
        )

        self.gnn_layers = nn.ModuleList()
        for _ in range(self.depth):
            # MLP cho message_nn (Bước 1 - D-MPNN style)
            # Input: cat(node_hid_feats, edge_hid_feats)
            # Output: node_hid_feats (để có thể cộng với (1+eps)x_i)
            message_mlp = nn.Sequential(
                nn.Linear(node_hid_feats + edge_hid_feats, node_hid_feats),
                nn.ReLU() # Activation bên trong message_nn
            )

            # MLP cho nn (MLP cập nhật cuối của GINE - Bước 3)
            # Input: node_hid_feats
            # Output: node_hid_feats
            gine_update_mlp = nn.Sequential(
                nn.Linear(node_hid_feats, node_hid_feats), # Có thể mở rộng ở giữa *2 ớ cuối
                nn.ReLU(),                                  # Activation
                nn.Linear(node_hid_feats, node_hid_feats) #*2 ở đầu
            )

            conv_layer = DMPNNInspiredGINEConv( # Sử dụng lớp GNN lai mới
                nn=gine_update_mlp,
                message_nn=message_mlp,
                train_eps=train_eps # Truyền train_eps vào
            )
            self.gnn_layers.append(conv_layer)

        # Phần Readout và Dropout (Giữ nguyên - Bước 5 GINE/Chuẩn)
        self.sparsify = nn.Sequential(
            nn.Linear(node_hid_feats, readout_feats), nn.PReLU() # Có PReLU (Activation)
        )
        self.dropout = nn.Dropout(dr)
        self.node_hid_feats = node_hid_feats # Lưu lại để lớp model có thể truy cập
        self.readout_feats = readout_feats   # Lưu lại

    def forward(self, data):
        node_feats_orig = data.x
        edge_feats_orig = data.edge_attr
        batch = data.batch
        edge_index = data.edge_index # Lấy từ data

        node_feats = self.project_node_feats(node_feats_orig)
        edge_feats = self.project_edge_feats(edge_feats_orig) # edge_feats giờ có size edge_hid_feats

        for i in range(self.depth):
            # Mỗi lớp gnn_layers[i] giờ là DMPNNInspiredGINEConv
            # Nó thực hiện Bước 1 (message D-MPNN), Bước 2 (aggregate GINE), Bước 3 (update GINE)
            node_feats = self.gnn_layers[i](node_feats, edge_index, edge_feats)

            # Bước 4: Activation giữa các lớp (Giữ kiểu GINE/Chuẩn)
            if i < self.depth - 1:
                node_feats = F.relu(node_feats) # Sử dụng F.relu thay vì nn.functional

            node_feats = self.dropout(node_feats)

        # Bước 5: Readout (Giữ kiểu GINE/Chuẩn)
        readout = global_add_pool(node_feats, batch)

        if self.readout_option:
            readout = self.sparsify(readout)

        return readout


#** THIẾT KẾ MODEL (Lớp Wrapper) **
import time
import json
import numpy as np
from torch.optim import Adam
from tqdm import tqdm
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score, root_mean_squared_error, mean_absolute_error
from pathlib import Path
import sys
import os


class model(nn.Module):
    def __init__(
        self,
        node_feat,
        edge_feat,
        out_dim=1,
        num_layer=3,
        node_hid_feats=300,
        edge_hid_feats=300,      # <<< THÊM THAM SỐ NÀY
        readout_feats=1024,
        predict_hidden_feats=512,
        readout_option=False,    # Sửa default cho khớp với GNN nếu muốn
        drop_ratio=0.1,
        train_eps: bool = False  # <<< THÊM THAM SỐ NÀY
    ):
        super(model, self).__init__()
        # emb_dim=1024 # Sẽ được xác định từ output của GNN
        
        self.gnn = GNN(
            node_in_feats=node_feat,
            edge_in_feats=edge_feat,
            depth=num_layer,
            node_hid_feats=node_hid_feats,
            edge_hid_feats=edge_hid_feats, # Truyền vào GNN
            readout_feats=readout_feats,
            dr=drop_ratio,
            readout_option=readout_option,
            train_eps=train_eps           # Truyền vào GNN
        )

        # Xác định kích thước đầu vào cho lớp predict dựa trên output của GNN
        if self.gnn.readout_option:
            gnn_output_feats = self.gnn.readout_feats
        else:
            gnn_output_feats = self.gnn.node_hid_feats


        self.predict = nn.Sequential(
            torch.nn.Linear(gnn_output_feats, predict_hidden_feats), # Sửa emb_dim
            torch.nn.PReLU(),
            torch.nn.Dropout(drop_ratio),
            torch.nn.Linear(predict_hidden_feats, predict_hidden_feats),
            torch.nn.PReLU(),
            torch.nn.Dropout(drop_ratio),
            torch.nn.Linear(predict_hidden_feats, out_dim),
        )

    def forward(self, mols): # data object from PyG
        graph_feats = self.gnn(mols) # mols is the PyG data object
        out = self.predict(graph_feats)
        return out

# --- Các hàm train và inference không cần thay đổi ---
def train(
    args,
    net,
    train_loader,
    val_loader,
    model_path,
    device,
    epochs=20,
    current_epoch=0,
    best_val_loss=1e10,
):
    monitor_path = args.monitor_folder + args.monitor_name
    n_epochs = epochs

    loss_fn = torch.nn.MSELoss()
    optimizer = Adam(net.parameters(), lr=5e-4, weight_decay=1e-5) ##default: lr=5e-4, weight_decay=1e-5

    for epoch in range(n_epochs):
        # training
        net.train()
        start_time = time.time()

        train_loss_list = []
        targets = []
        preds = []

        for batchdata in tqdm(train_loader, desc="Training"):
            batchdata=batchdata.to(device)
            pred = net(batchdata)
            # print(pred.shape)
            labels = batchdata.y
            # print(labels.shape)
            # assert 1==2
            targets.extend(labels.tolist())
            labels = labels.to(device)

            preds.extend(pred.tolist())
            loss = loss_fn(pred.view(-1), labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss = loss.detach().item()
            train_loss_list.append(train_loss)

        rmse = root_mean_squared_error(targets, preds)
        mae = mean_absolute_error(targets, preds)
        print(
            "--- training epoch %d, loss %.3f, rmse %.3f, mae %.3f, time elapsed(min) %.2f---"
            % (
                epoch,
                np.mean(train_loss_list),
                rmse,
                mae,
                (time.time() - start_time) / 60,
            )
        )

        # validation
        net.eval()
        val_rmse, val_mae, val_loss = inference(args, net, val_loader, device, loss_fn)

        print(
            "--- validation at epoch %d, val_loss %.3f, val_rmse %.3f, val_mae %.3f ---"
            % (epoch, val_loss, val_rmse, val_mae)
        )
        print("\n" + "*" * 100)

        dict = {
            "epoch": epoch + current_epoch,
            "train_loss": np.mean(train_loss_list),
            "val_loss": val_loss,
            "train_rmse": rmse,
            "val_rmse": val_rmse,
        }
        with open(monitor_path, "a") as f:
            f.write(json.dumps(dict) + "\n")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch + current_epoch,
                    "model_state_dict": net.state_dict(),
                    "val_loss": best_val_loss,
                },
                model_path,
            )


def inference(args, net, test_loader, device, loss_fn=None):
    # batch_size = test_loader.batch_size

    net.eval()
    inference_loss_list = []
    preds = []
    targets = []

    with torch.no_grad():
        for batchdata in tqdm(test_loader, desc="Testing"):
            batchdata=batchdata.to(device)
            pred = net(batchdata)
            labels = batchdata.y
            targets.extend(labels.tolist())
            labels = labels.to(device)

            preds.extend(pred.tolist())

            if loss_fn is not None:
                inference_loss = loss_fn(pred.view(-1), labels)
                inference_loss_list.append(inference_loss.item())

    rmse = root_mean_squared_error(targets, preds)
    mae = mean_absolute_error(targets, preds)

    if loss_fn is None:
        return rmse, mae
    else:
        return rmse, mae, np.mean(inference_loss_list)
