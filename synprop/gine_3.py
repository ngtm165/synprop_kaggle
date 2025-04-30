import torch
import torch.nn as nn
import torch.nn.functional as F # Import F cho ReLU giữa các lớp
# from torch_geometric.nn.conv import GINEConv # Không dùng GINEConv nữa
from torch_geometric.nn.pool import global_add_pool
from torch_geometric.nn import MessagePassing # Import lại nếu cần định nghĩa lớp GNN mới ở đây

# --- Định nghĩa lớp GNN lai mới: DMPNN Message & DMPNN Update ---
#    (Lớp này thực hiện Bước 1, 2, 3 theo yêu cầu)
class DMPNN_Hybrid_Conv(MessagePassing):
    """
    Lớp GNN lấy cảm hứng từ D-MPNN cho cả bước tạo message và bước cập nhật nút.
    Message m_{j->i} = message_nn(cat(x_j, edge_attr))  (Bước 1 - D-MPNN style)
    Aggregate: sum                                       (Bước 2 - D-MPNN/GINE style)
    Update x'_i = update_nn(cat(x_i, sum(m_{j->i})))     (Bước 3 - D-MPNN style)
    """
    def __init__(self, message_nn: torch.nn.Module, update_nn: torch.nn.Module,
                 aggr: str = 'sum', **kwargs):
        super().__init__(aggr=aggr, **kwargs)
        self.message_nn = message_nn
        self.update_nn = update_nn
        self.reset_parameters()

    def reset_parameters(self):
        # Reset các tham số cho MLP
        def _reset_module(module):
            for layer in module:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        _reset_module(self.message_nn)
        _reset_module(self.update_nn)


    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        # Bước 1 & 2: Truyền và tổng hợp thông điệp
        aggregated_messages = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        # Bước 3: Cập nhật nút kiểu D-MPNN
        update_input = torch.cat([x, aggregated_messages], dim=-1)
        out = self.update_nn(update_input) # MLP update_nn chứa Activation (Bước 4) bên trong
        return out

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        # Bước 1: Tạo thông điệp kiểu D-MPNN
        input_message = torch.cat([x_j, edge_attr], dim=-1)
        # message_nn chứa Activation (Bước 4) bên trong
        return self.message_nn(input_message)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(message_nn={self.message_nn}, update_nn={self.update_nn})'

# --- Lớp GNN tổng thể ĐÃ ĐƯỢC MODIFY ---
class GNN(nn.Module):
    def __init__(
        self,
        node_in_feats,
        edge_in_feats,
        depth=5,
        node_hid_feats=300,
        edge_hid_feats=300, # Thêm tham số này
        readout_feats=1024,
        dr=0.1,
        readout_option=True,
    ):
        super(GNN, self).__init__()

        self.depth = depth
        self.readout_option = readout_option # Lưu option

        # --- Các lớp chiếu đặc trưng ban đầu (Giữ nguyên) ---
        self.project_node_feats = nn.Sequential(
            nn.Linear(node_in_feats, node_hid_feats), nn.ReLU() # Có ReLU - Bước 4
        )
        self.project_edge_feats = nn.Sequential(
            nn.Linear(edge_in_feats, edge_hid_feats) # Có thể thêm ReLU nếu muốn
        )

        # --- Tạo danh sách các lớp GNN MỚI (DMPNN_Hybrid_Conv) ---
        self.gnn_layers = nn.ModuleList()
        for _ in range(self.depth):
            # Định nghĩa MLP cho message_nn (Bước 1 D-MPNN)
            message_mlp = nn.Sequential(
                nn.Linear(node_hid_feats + edge_hid_feats, node_hid_feats),
                nn.ReLU() # Bước 4: Activation bên trong message MLP
            )
            # Định nghĩa MLP cho update_nn (Bước 3 D-MPNN)
            update_mlp = nn.Sequential(
                nn.Linear(node_hid_feats * 2, node_hid_feats * 2), # Input: cat(x, agg_msg)
                nn.ReLU(), # Bước 4: Activation bên trong update MLP
                nn.Linear(node_hid_feats * 2, node_hid_feats)
            )
            # Tạo lớp GNN lai mới (Thực hiện Bước 1, 2, 3)
            conv_layer = DMPNN_Hybrid_Conv(
                message_nn=message_mlp,
                update_nn=update_mlp,
                aggr='sum' # Bước 2 D-MPNN/GINE (sum aggregation)
            )
            self.gnn_layers.append(conv_layer)

        # --- Phần Readout và Dropout (Giữ nguyên - Bước 5 GINE/Chuẩn) ---
        self.sparsify = nn.Sequential(
            nn.Linear(node_hid_feats, readout_feats), nn.PReLU() # Có PReLU - Bước 4
        )
        self.dropout = nn.Dropout(dr)
        # Lưu lại các thông số khác nếu cần
        self.node_hid_feats = node_hid_feats
        self.readout_feats = readout_feats


    def forward(self, data):
        node_feats_orig = data.x
        edge_feats_orig = data.edge_attr
        batch = data.batch
        edge_index = data.edge_index # Lấy edge_index từ data

        # Chiếu đặc trưng ban đầu
        node_feats = self.project_node_feats(node_feats_orig)
        edge_feats = self.project_edge_feats(edge_feats_orig)

        # Vòng lặp qua các lớp GNN mới (Mỗi lớp thực hiện Bước 1, 2, 3)
        for i in range(self.depth):
            node_feats = self.gnn_layers[i](node_feats, edge_index, edge_feats)

            # --- Bước 4: Activation (Giữ kiểu GINE/Chuẩn) ---
            # Áp dụng ReLU giữa các lớp (trừ lớp cuối trước readout)
            if i < self.depth - 1:
                 node_feats = F.relu(node_feats) # Sử dụng F.relu

            # Có thể giữ lại dropout
            node_feats = self.dropout(node_feats)

        # --- Bước 5: Readout (Giữ kiểu GINE/Chuẩn) ---
        readout = global_add_pool(node_feats, batch) # Pooling

        if self.readout_option:
            readout = self.sparsify(readout) # MLP readout cuối (có Activation PReLU)

        return readout