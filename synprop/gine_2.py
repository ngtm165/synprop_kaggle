import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch_geometric.nn.pool import global_add_pool
from torch_geometric.nn import MessagePassing 

# --- Định nghĩa lớp GNN lai mới: DMPNN Message & DMPNN Update ---
class DMPNN_Hybrid_Conv(MessagePassing): # Đổi tên lớp ##BƯỚC 1,3 D-MPNN
    """
    Lớp GNN lấy cảm hứng từ D-MPNN cho cả bước tạo message và bước cập nhật nút.
    Message m_{j->i} = message_nn(cat(x_j, edge_attr))
    Update x'_i = update_nn(cat(x_i, sum(m_{j->i}))) <-- THAY ĐỔI UPDATE
    """
    # Bỏ eps, train_eps. Đổi nn -> update_nn
    def __init__(self, message_nn: torch.nn.Module, update_nn: torch.nn.Module, 
                 aggr: str = 'sum', **kwargs): # Đổi aggr='sum' thành 'add' cho nhất quán PyG
        super().__init__(aggr=aggr, **kwargs) 
        self.message_nn = message_nn 
        self.update_nn = update_nn # MLP cập nhật cuối cùng theo kiểu D-MPNN
        
        # Không còn eps
        
        self.reset_parameters()

    def reset_parameters(self):
        # Reset MLP tạo message
        # Sửa lại cách duyệt module để reset tốt hơn
        for module in self.message_nn.modules():
             if hasattr(module, 'reset_parameters'):
                 module.reset_parameters()
        # Reset MLP cập nhật
        for module in self.update_nn.modules(): # Đổi nn thành update_nn
             if hasattr(module, 'reset_parameters'):
                 module.reset_parameters()
        # Không còn reset eps

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Thực hiện quá trình truyền thông điệp và cập nhật nút kiểu D-MPNN.
        """
        # Bước 1 & 2: Truyền và tổng hợp thông điệp
        aggregated_messages = self.propagate(edge_index, x=x, edge_attr=edge_attr) 
        
        # --- THAY ĐỔI BƯỚC 3: CẬP NHẬT NÚT ---
        # Kết hợp thông tin nút gốc và thông điệp tổng hợp bằng cat
        # update_input = torch.cat([x, aggregated_messages], dim=-1) #mang 2
        update_input = x + aggregated_messages #mang 2.1
        print(update_input)

        # Kích thước đầu vào cho update_nn là node_hid_feats * 2

        # Đưa qua MLP cập nhật cuối cùng
        out = self.update_nn(update_input) 

        return out

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Tạo thông điệp m_{j->i} = message_nn(cat(x_j, edge_attr))
        (Giữ nguyên như lớp custom trước)
        """
        input_message = torch.cat([x_j, edge_attr], dim=-1) 
        return self.message_nn(input_message) 
    
    # Bỏ hàm __repr__ cũ đi, thay bằng hàm mới
    def __repr__(self) -> str: 
        return f'{self.__class__.__name__}(message_nn={self.message_nn}, update_nn={self.update_nn})' # Cập nhật repr

# --- Tinh chỉnh lớp GNN của bạn ---
class GNN(nn.Module):
    def __init__(
        self,
        node_in_feats,
        edge_in_feats,
        depth=5, 
        node_hid_feats=300, 
        edge_hid_feats=300, 
        readout_feats=1024,
        dr=0.1, 
        readout_option=True,
        # train_eps: bool = False # Bỏ train_eps vì lớp GNN mới không dùng
    ):
        super(GNN, self).__init__()

        self.depth = depth
        self.readout_option = readout_option

        # Project đặc trưng nút đầu vào -> kích thước ẩn (Giữ nguyên)
        self.project_node_feats = nn.Sequential(
            nn.Linear(node_in_feats, node_hid_feats), nn.ReLU()
        )

        # Project đặc trưng cạnh đầu vào -> kích thước ẩn (Giữ nguyên)
        self.project_edge_feats = nn.Sequential(
            nn.Linear(edge_in_feats, edge_hid_feats) 
        )

        # Tạo danh sách các lớp GNN mới
        self.gnn_layers = nn.ModuleList()
        for _ in range(self.depth):
            # --- THAY ĐỔI CÁCH ĐỊNH NGHĨA MLPS CHO LỚP GNN MỚI ---
            
            # 1. MLP W_msg (message_nn) - Giữ nguyên định nghĩa
            message_mlp = nn.Sequential(
                nn.Linear(node_hid_feats + edge_hid_feats, node_hid_feats), 
                nn.ReLU() 
            )

            # 2. MLP cập nhật cuối (update_nn) - THAY ĐỔI ĐỊNH NGHĨA
            # Đầu vào là cat(x, aggregated_messages) -> node_hid_feats * 2
            # Đầu ra là node_hid_feats
            update_mlp = nn.Sequential(
                nn.Linear(node_hid_feats, node_hid_feats), # Input: node_hid_feats * 2 ## mang 2: (node_hid_feats * 2, node_hid_feats * 2), mang 2.1 bỏ hết *2
                nn.ReLU(),
                nn.Linear(node_hid_feats, node_hid_feats)  # Output: node_hid_feats  ##mang 2: (node_hid_feats * 2, node_hid_feats), mang 2.1 bỏ hết *2
            )

            # 3. Tạo lớp GNN lai MỚI
            conv_layer = DMPNN_Hybrid_Conv( # Gọi lớp GNN mới
                message_nn=message_mlp, 
                update_nn=update_mlp       # Truyền MLP cập nhật mới
                # Không còn nn=mlp_h_theta, không còn eps, train_eps
            )
            self.gnn_layers.append(conv_layer)


        # Các lớp còn lại giữ nguyên
        self.sparsify = nn.Sequential(
            nn.Linear(node_hid_feats, readout_feats), nn.PReLU()
        )
        self.dropout = nn.Dropout(dr)
        # self.readout_option = readout_option # Đã gán ở trên
        self.node_hid_feats = node_hid_feats 
        self.readout_feats = readout_feats   

    def forward(self, data):
        # Phần forward của GNN tổng thể giữ nguyên cấu trúc
        node_feats_orig = data.x
        edge_feats_orig = data.edge_attr
        batch = data.batch
        edge_index = data.edge_index 

        node_feats = self.project_node_feats(node_feats_orig)
        edge_feats = self.project_edge_feats(edge_feats_orig) 

        # node_feats_list = [node_feats] # Bỏ nếu không dùng skip connection
        for i in range(self.depth):
            # Chỉ cần gọi lớp GNN mới, nó sẽ tự xử lý bên trong
            node_feats = self.gnn_layers[i](node_feats, edge_index, edge_feats) 
            
            # Có thể vẫn áp dụng dropout giữa các lớp
            node_feats = self.dropout(node_feats)
            # node_feats_list.append(node_feats) # Bỏ nếu không dùng skip connection

        readout = global_add_pool(node_feats, batch)

        if self.readout_option:
            readout = self.sparsify(readout)

        return readout
