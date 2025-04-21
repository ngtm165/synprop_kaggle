import torch
import torch.nn as nn
import torch.nn.functional as F # Thêm import này
from torch_geometric.nn.pool import global_add_pool
from torch_geometric.nn import MessagePassing # Import lớp cơ sở MessagePassing

# --- Định nghĩa lớp DMPNNInspiredGINEConv mới ---
class DMPNNInspiredGINEConv(MessagePassing):
    """
    Lớp GNN lấy cảm hứng từ D-MPNN (bước tạo message) và GINE (cấu trúc cập nhật nút).
    Message m_{j->i} = ReLU(Linear(cat(x_j, edge_attr)))
    Update x'_i = MLP((1+eps)x_i + sum(m_{j->i}))
    """
    def __init__(self, nn: torch.nn.Module, message_nn: torch.nn.Module, 
                 eps: float = 0., train_eps: bool = False,
                 aggr: str = 'sum', **kwargs):
        super().__init__(aggr=aggr, **kwargs) 
        self.nn = nn # MLP h_theta cuối cùng, giống GINEConv
        self.message_nn = message_nn # Mạng nơ-ron để tạo message từ cat(x_j, edge_attr)
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        
        # Reset parameters của các mạng nơ-ron con (quan trọng)
        self.reset_parameters()

    def reset_parameters(self):
        # Reset MLP cuối
        for module in self.nn:
             if hasattr(module, 'reset_parameters'):
                 module.reset_parameters()
        # Reset MLP tạo message
        for module in self.message_nn:
             if hasattr(module, 'reset_parameters'):
                 module.reset_parameters()
        # Reset epsilon nếu học được
        if hasattr(self.eps, 'data'):
            self.eps.data.fill_(self.initial_eps)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Thực hiện quá trình truyền thông điệp và cập nhật nút.
        """
        # x có shape [N, node_hid_feats]
        # edge_index có shape [2, E]
        # edge_attr có shape [E, node_hid_feats] (sau khi project)

        # Bước 1 & 2: Truyền và tổng hợp thông điệp (propagate -> aggregate)
        # self.propagate sẽ gọi self.message bên dưới
        aggregated_messages = self.propagate(edge_index, x=x, edge_attr=edge_attr) 
        # aggregated_messages có shape [N, node_hid_feats] (output của message_nn)

        # Bước 3: Cập nhật nút (Combine và Transformation cuối bởi self.nn)
        # Kết hợp thông tin nút gốc và thông điệp tổng hợp
        out = (1 + self.eps) * x + aggregated_messages

        # Đưa qua MLP cuối cùng (h_theta)
        out = self.nn(out)

        return out

    # def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
    #     """
    #     Tạo thông điệp m_{j->i} = ReLU(Linear(cat(x_j, edge_attr)))
    #     x_j shape: [E, node_hid_feats] (đặc trưng nút nguồn j cho mỗi cạnh)
    #     edge_attr shape: [E, node_hid_feats] (đặc trưng cạnh tương ứng)
    #     """
    #     # Ghép nối đặc trưng nút nguồn và đặc trưng cạnh
    #     # Kích thước đầu vào cho message_nn sẽ là node_hid_feats + node_hid_feats
    #     input_message = torch.cat([x_j, edge_attr], dim=-1) 
        
    #     # Đưa qua mạng nơ-ron tạo message và áp dụng ReLU
    #     # Giả sử message_nn đã bao gồm cả ReLU hoặc bạn có thể thêm ở đây
    #     # Ví dụ: return F.relu(self.message_nn(input_message)) 
    #     # Nếu message_nn đã có ReLU rồi thì chỉ cần:
    #     return self.message_nn(input_message) 
    
    def message(self, edge_attr: torch.Tensor) -> torch.Tensor: # Không cần x_j ở đây nữa
    # edge_attr bây giờ đã là cat(nút_nguồn_ban_đầu, cạnh_gốc)
        return self.message_nn(edge_attr)

    def __repr__(self) -> str: ##thử bỏ
        return f'{self.__class__.__name__}(nn={self.nn}, message_nn={self.message_nn})'

# --- Tinh chỉnh lớp GNN của bạn ---
class GNN(nn.Module):
    def __init__(
        self,
        node_in_feats,
        edge_in_feats,
        depth=5, 
        node_hid_feats=300, 
        edge_hid_feats=300, # Thêm tham số này nếu muốn kích thước ẩn cạnh khác nút
        readout_feats=1024,
        dr=0.1, 
        readout_option=True,
        train_eps: bool = False # Thêm tùy chọn học epsilon
    ):
        super(GNN, self).__init__()

        self.depth = depth

        # Project đặc trưng nút đầu vào -> kích thước ẩn
        self.project_node_feats = nn.Sequential(
            nn.Linear(node_in_feats, node_hid_feats), nn.ReLU()
        )

        # Project đặc trưng cạnh đầu vào -> kích thước ẩn (ví dụ: cùng kích thước với nút)
        # Quan trọng: kích thước này (edge_hid_feats) sẽ được dùng trong message_nn
        self.project_edge_feats = nn.Sequential(
            nn.Linear(edge_in_feats, edge_hid_feats) 
        )

        # Tạo danh sách các lớp GNN mới
        self.gnn_layers = nn.ModuleList()
        for _ in range(self.depth):
            # MLP h_theta cuối cùng (nhận input là node_hid_feats)
            mlp_h_theta = nn.Sequential(
                nn.Linear(node_hid_feats, node_hid_feats), # Có thể tăng chiều ở giữa, THỬ BỎ *2 CHO NODE_HID_FEATS SAU
                nn.ReLU(),
                nn.Linear(node_hid_feats, node_hid_feats) #THỬ BỎ *2 CHO NODE_HID_FEATS TRƯỚC
            )
            
            # MLP W_msg (message_nn) nhận input là cat(x_j, edge_attr) 
            # Kích thước input: node_hid_feats + edge_hid_feats
            # Kích thước output: node_hid_feats (để cộng được với (1+eps)x_i)
            message_mlp = nn.Sequential(
                nn.Linear(node_hid_feats + edge_hid_feats, node_hid_feats), # Quan trọng kích thước input
                nn.ReLU() # Thêm ReLU ở đây hoặc trong hàm message
            )

            # Tạo lớp GNN lai
            conv_layer = DMPNNInspiredGINEConv(
                nn=mlp_h_theta, 
                message_nn=message_mlp,
                train_eps=train_eps 
                # eps có thể để mặc định là 0 hoặc truyền giá trị khác
            )
            self.gnn_layers.append(conv_layer)


        # Các lớp còn lại giữ nguyên
        self.sparsify = nn.Sequential(
            nn.Linear(node_hid_feats, readout_feats), nn.PReLU()
        )
        self.dropout = nn.Dropout(dr)
        self.readout_option = readout_option # Lưu lại tùy chọn readout
        self.node_hid_feats = node_hid_feats # Lưu lại kích thước ẩn nút
        self.readout_feats = readout_feats   # Lưu lại kích thước readout

    def forward(self, data):
        node_feats_orig = data.x
        edge_feats_orig = data.edge_attr
        batch = data.batch
        edge_index = data.edge_index # Lấy edge_index

        # Project đặc trưng ban đầu
        node_feats = self.project_node_feats(node_feats_orig)
        edge_feats = self.project_edge_feats(edge_feats_orig) # Bây giờ edge_feats có kích thước edge_hid_feats

        # Vòng lặp qua các lớp GNN (giờ là lớp DMPNNInspiredGINEConv)
        node_feats_list = [node_feats] # Lưu lại node_feats các lớp (có thể hữu ích cho skip connection)
        for i in range(self.depth):
            node_feats = self.gnn_layers[i](node_feats, edge_index, edge_feats)
            
            # Có thể giữ lại ReLU và Dropout giữa các lớp GNN
            # if i < self.depth - 1: # Không áp dụng ReLU cho lớp cuối cùng trước readout
            #     node_feats = F.relu(node_feats) # Đảm bảo đã import torch.nn.functional as F
            
            # Skip connection (ví dụ cộng): có thể thêm nếu muốn
            # node_feats = node_feats + node_feats_list[-1] 
            # node_feats_list.append(node_feats)

            node_feats = self.dropout(node_feats)

        # Readout giữ nguyên
        readout = global_add_pool(node_feats, batch)

        if self.readout_option:
            readout = self.sparsify(readout)

        return readout