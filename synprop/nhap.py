from rdkit import Chem
from torch_geometric.data import Data, Dataset
import pandas as pd
import numpy as np
import torch
import sys
import os
from pathlib import Path
import pickle
import gzip
import random
import networkx as nx 

root_dir = str(Path(__file__).resolve().parents[1])
os.chdir(root_dir)

atom_list = list(range(1, 119))
charge_list = [-2, -1, 0, 1, 2, 'other']
hybridization = ['SP', 'SP2', 'SP3', 'other']
valence = [1, 2, 3, 4, 5, 6, 'other']

bond_type1 = [1, 2, 3, 'other']
bond_type2 = ['normal', 'increase', 'decrease', 'other']

def read_data(data_path, graph_path, target):
    graph, labels = [], []
    data = pd.read_csv(data_path)
    labels_lst = data[target].tolist()  # Directly get the target column as a list
    with gzip.open(graph_path, 'rb') as f:
        graphs = pickle.load(f)
    # graphs_lst = [i['ITSGraph'][2] for i in graphs]
    graphs_lst = list(graphs.values())  # Chuyển đổi values() thành list
    
    return graphs_lst, labels_lst


class ReactionDataset(Dataset):
    def __init__(self, data_path, graph_path, target):
        super(Dataset, self).__init__()
        self.graph, self.labels = read_data(data_path, graph_path, target)

    def __getitem__(self, index):
        graph = self.graph[index]
        lst_nodes = list(graph.nodes())
        lst_nodes_update = [lst_nodes.index(i) for i in lst_nodes]
        lst_edges = list(graph.edges())
        lst_edges_update = [(lst_nodes.index(u), lst_nodes.index(v)) for u, v in lst_edges]
        label = self.labels[index]

        #atom features
        pt = Chem.GetPeriodicTable()
        # atom_fea_graph = []
        max_neighbors = 6  # Tìm số neighbors lớn nhất. viết kèm 3 dòng dưới
        
        atom_fea_graph = []
        atom_hybrid_change = []
        hcount_change = []
        hybrid_change = []


        
        for i in lst_nodes:
            
            atom_data = graph.nodes(data=True)[i] #ver 7_mới

            #charge
            charge_1 = atom_data['typesGH'][0][3] 
            charge_2 = atom_data['typesGH'][1][3]
            charge_atom = [charge_1] + [charge_2] #ver_3
            charge_change = charge_1 - charge_2
        
            
            atom_fea = quantum_features + e_max + [charge_1] + [charge_change] + h_1 + hcount_change + atom_hybrid_p_1 + atom_hybrid_change + [neighbor_count_1] + [neighbor_change] #5.1
            # atom_fea = quantum_features + e_max + atom_hybrid_p_1 + atom_hybrid_change + [neighbor_count_1] + [neighbor_change]  #5.2 

            atom_fea_graph.append(atom_fea)

        
        #bond_feature
        row, col = [], []
        edge_feat_graph = []
        reaction_center_data = []

        for idx, bond in enumerate(lst_edges_update): # bond là cặp (chỉ số nút u, chỉ số nút v)
            u = bond[0] # Chỉ số nút nguồn cho hướng u -> v
            v = bond[1] # Chỉ số nút nguồn cho hướng v -> u

            row += [u, v]
            col += [v, u]

            # # Thêm các đặc trưng cạnh mới
            order_0, order_1 = list(graph.edges(data=True))[idx][2]['order']
            standard_order = list(graph.edges(data=True))[idx][2]['standard_order']
            
            changes = []

            edge_fea = edge_fea1 + edge_fea2 

            # Lấy đặc trưng của hai nguyên tử tham gia liên kết
            atom_feat_u = atom_fea_graph[u] 
            atom_feat_v = atom_fea_graph[v]

            # Tạo đặc trưng có hướng cho u -> v: Ghép đặc trưng nguyên tử nguồn u với đặc trưng liên kết
            directed_feature_uv = np.concatenate((atom_feat_u, edge_fea)).tolist() # Ví dụ dùng numpy concatenate
            edge_feat_graph.append(directed_feature_uv)

            # Tạo đặc trưng có hướng cho v -> u: Ghép đặc trưng nguyên tử nguồn v với đặc trưng liên kết
            directed_feature_vu = np.concatenate((atom_feat_v, edge_fea)).tolist() # Ví dụ dùng numpy concatenate
            edge_feat_graph.append(directed_feature_vu)

            if any(x != 0 for x in changes):
                bond_info = {
                    'edge_indices': {(u,v),(v,u)},
                    'edge_features': edge_fea,
                    'node_features': {
                        u: atom_fea_graph[u],
                        v: atom_fea_graph[v]
                    }
                }
                reaction_center_data.append(bond_info)
                    

                
        #RC
        RC_node_features = {}
        RC_edge_features = []
        RC_edge_index_row = []
        RC_edge_index_col = []
        atom_fea_graph_RC, edge_feat_graph_RC = [], []

        num_original_atoms = len(lst_nodes_update)

        # Lấy ra tập hợp các chỉ số nút duy nhất từ RC
        unique_rc_nodes = set()
        for bond_data in reaction_center_data:
            unique_rc_nodes.update(bond_data['node_features'].keys())
            

        # Sắp xếp để đảm bảo thứ tự mapping luôn nhất quán
        sorted_rc_nodes = sorted(list(unique_rc_nodes))

        node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_rc_nodes, start=num_original_atoms)}

        
        reindexed_rc_data = []
        for bond_data in reaction_center_data:
            # Lấy ra các chỉ số cũ
            u_old, v_old = bond_data['node_features'].keys()
            
            u_new = node_mapping[u_old]
            v_new = node_mapping[v_old]
            
            # Tạo một dictionary mới với các thông tin đã được re-index
            new_bond_info = {
                'edge_indices': {(u_new, v_new), (v_new, u_new)},
                'edge_features': bond_data['edge_features'],
                'node_features': {
                    u_new: bond_data['node_features'][u_old],
                    v_new: bond_data['node_features'][v_old]
                }
            }
            reindexed_rc_data.append(new_bond_info)
        unique_node_features = {}
        # 1. Lặp qua từng 'gói dữ liệu' trong danh sách chính
        for bond_data in reindexed_rc_data:
            RC_node_features.update(bond_data['node_features'])
            unique_node_features.update(bond_data['node_features'])

            RC_edge_features = bond_data['edge_features']
            RC_index = bond_data['edge_indices']
            u_new, v_new = list(bond_data['node_features'].keys()) # Lấy 2 nút mới
   
            # Lấy đặc trưng tương ứng của 2 nút này
            atom_feat_rc_u = bond_data['node_features'][u_new]
            atom_feat_rc_v = bond_data['node_features'][v_new]
            # BƯỚC THAY ĐỔI: TẠO ĐẶC TRƯNG CẠNH CÓ HƯỚNG CHO RC
            directed_feature_rc_uv = atom_feat_rc_u + RC_edge_features
            edge_feat_graph_RC.append(directed_feature_rc_uv)
            directed_feature_rc_vu = atom_feat_rc_v + RC_edge_features
            edge_feat_graph_RC.append(directed_feature_rc_vu)
            
            RC_edge_index_row.append(min(RC_index))
            RC_edge_index_col.append(max(RC_index))

        # 2. Lấy ra danh sách các feature (values) từ dictionary kết quả
        node_RC = [unique_node_features[n_idx] for n_idx in node_mapping.values()]
        atom_fea_graph_RC.extend(node_RC)
        RC_row, RC_col = [], []

        for i in RC_edge_index_row:
            RC_row.extend(i)
        for a in RC_edge_index_col: 
            RC_col.extend(a)
        row_RC += RC_row
        col_RC += RC_col
        
        # supernode
        supernode_index = len(atom_fea_graph) 
        supernode_node_feature = [0.0]*23
        supernode_edge_feature = [0.0]*6
        atom_fea_graph_super, edge_feat_graph_super = [], []
        row_super, col_super = [], []

        atom_fea_graph_super.append(supernode_node_feature)

        for rc_node_old_index in (list(unique_rc_nodes)):
            rc_node_new_index = node_mapping[rc_node_old_index]
            rc_node_feature = RC_node_features[rc_node_new_index]
            
            # BƯỚC THAY ĐỔI: TẠO ĐẶC TRƯNG CÓ HƯỚNG
            directed_feat_rc_to_super = rc_node_feature + supernode_edge_feature             # Hướng từ nút RC -> supernode
            edge_feat_graph_super.append(directed_feat_rc_to_super)
            directed_feat_super_to_rc = supernode_node_feature + supernode_edge_feature      # Hướng từ supernode -> nút RC
            edge_feat_graph_super.append(directed_feat_super_to_rc)
            
            row_super.append(rc_node_new_index)
            col_super.append(supernode_index)
            row_super.append(supernode_index)
            col_super.append(rc_node_new_index)


        edge_index=torch.tensor([row,col])
        edge_attr=torch.tensor(np.array(edge_feat_graph),dtype=torch.float)
        node_attr=torch.tensor(np.array(atom_fea_graph),dtype=torch.float)
        y=torch.tensor(label,dtype=torch.float)
        data= Data(x=node_attr,y=y,edge_index=edge_index,edge_attr=edge_attr) 
        
        edge_index_RC=torch.tensor([row_RC,col_RC])
        edge_attr_RC=torch.tensor(np.array(edge_feat_graph_RC),dtype=torch.float)
        node_attr_RC=torch.tensor(np.array(atom_fea_graph_RC),dtype=torch.float)
        y_RC=torch.tensor(label,dtype=torch.float)
        data_RC= Data(x=node_attr_RC,y=y_RC,edge_index=edge_index_RC,edge_attr=edge_attr_RC) 
        
        edge_index_super=torch.tensor([row_super,col_super])
        edge_attr_super=torch.tensor(np.array(edge_feat_graph_super),dtype=torch.float)
        node_attr_super=torch.tensor(np.array(atom_fea_graph_super),dtype=torch.float)
        y_super=torch.tensor(label,dtype=torch.float)
        data_super= Data(x=node_attr_super,y=y_super,edge_index=edge_index_super,edge_attr=edge_attr_super) 
        
        return data, data_RC, data_super

    def __len__(self):
        return len(self.graph)

def main():
    folder_list = ['test', 'val', 'train']
    for i in folder_list:
        data_path=f'./Data/regression/barriers_cycloadd/{i}.csv'
        graph_path=f'./Data/regression/barriers_cycloadd/its_new/barriers_cycloadd_aam_{i}.pkl.gz'
        target='G_act'
        graphdata=ReactionDataset(data_path,graph_path,target)
        print(graphdata.__getitem__(8))


            # --- Thu thập dữ liệu từ Dataset ---
        all_edge_indices = []
        all_edge_attrs = []
        all_node_attrs = []
        all_ys = []
        output_filename = f'barriers_cycloadd_aam_{i}_processed_data.npz'
        
            # Tạo đường dẫn file output hoàn chỉnh
        
        output_folder = './output/super/barriers_cycloadd'
        os.makedirs(output_folder, exist_ok=True) 
        output_npz_file = os.path.join(output_folder, output_filename)

        print("\nĐang trích xuất dữ liệu từ từng mẫu trong dataset...")
        for i in range(len(graphdata)):
            try:
                data_item = graphdata[i]
                
                # Chuyển tensor sang NumPy array để lưu
                all_edge_indices.append(data_item.edge_index.numpy())
                all_edge_attrs.append(data_item.edge_attr.numpy())
                all_node_attrs.append(data_item.x.numpy()) # node_attr là data_item.x
                all_ys.append(data_item.y.numpy())         # y là data_item.y
                
                print(f"  Đã xử lý mẫu {i+1}/{len(graphdata)}")
                print(f"    Node features shape: {data_item.x.shape}")
                print(f"    Edge index shape: {data_item.edge_index.shape}")
                print(f"    Edge features shape: {data_item.edge_attr.shape}")
                print(f"    Label shape: {data_item.y.shape}")

            except Exception as e:
                print(f"Lỗi khi xử lý mẫu {i}: {e}")
                # Quyết định xem có nên bỏ qua mẫu lỗi hay dừng hẳn
                # continue 
                # raise
        #----------------------------------------------------
        # 1. Tạo mảng object tường minh cho edge_indices
        if all_edge_indices: # Kiểm tra nếu danh sách không rỗng
            num_samples_ei = len(all_edge_indices)
            edge_indices_to_save = np.empty(num_samples_ei, dtype=object)
            for i in range(num_samples_ei):
                edge_indices_to_save[i] = all_edge_indices[i]
        else:
            edge_indices_to_save = np.array([], dtype=object)
        
        # 2. Tạo mảng object tường minh cho edge_attrs
        if all_edge_attrs: # Kiểm tra nếu danh sách không rỗng
            num_samples_ea = len(all_edge_attrs)
            edge_attrs_to_save = np.empty(num_samples_ea, dtype=object)
            for i in range(num_samples_ea):
                edge_attrs_to_save[i] = all_edge_attrs[i]
        else:
            edge_attrs_to_save = np.array([], dtype=object)
        
        # 3. Tạo mảng object tường minh cho node_attrs
        if all_node_attrs: # Kiểm tra nếu danh sách không rỗng
            num_samples_na = len(all_node_attrs)
            node_attrs_to_save = np.empty(num_samples_na, dtype=object)
            for i in range(num_samples_na):
                node_attrs_to_save[i] = all_node_attrs[i]
        else:
            node_attrs_to_save = np.array([], dtype=object)
        
        # 4. Tạo mảng object tường minh cho ys
        if all_ys: # Kiểm tra nếu danh sách không rỗng
            num_samples_y = len(all_ys)
            ys_to_save = np.empty(num_samples_y, dtype=object)
            for i in range(num_samples_y):
                ys_to_save[i] = all_ys[i]
        else:
            ys_to_save = np.array([], dtype=object)
        
        # --- Lưu vào file .npz sử dụng các mảng object đã tạo tường minh ---
        print(f"\nĐang lưu dữ liệu vào file: {output_npz_file}")
        np.savez_compressed(output_npz_file,
                            edge_indices=edge_indices_to_save,
                            edge_attrs=edge_attrs_to_save,
                            node_attrs=node_attrs_to_save,
                            ys=ys_to_save)
        
        print(f"Đã lưu thành công dữ liệu vào '{output_npz_file}'.")
        
        # --- Cách tải lại dữ liệu (ví dụ) ---
        print("\n--- Ví dụ cách tải lại dữ liệu ---")
        loaded_data = np.load(output_npz_file, allow_pickle=True)
        
        # Truy cập dữ liệu đã tải
        loaded_edge_indices = loaded_data['edge_indices']
        loaded_edge_attrs = loaded_data['edge_attrs']
        loaded_node_attrs = loaded_data['node_attrs']
        loaded_ys = loaded_data['ys']
        
        if len(loaded_node_attrs) > 0:
            print(f"Số lượng mẫu đã tải: {len(loaded_node_attrs)}")
            print(f"  Shape của node_attrs của mẫu đầu tiên: {loaded_node_attrs[0].shape if hasattr(loaded_node_attrs[0], 'shape') else type(loaded_node_attrs[0])}")
            print(f"  Shape của edge_indices của mẫu đầu tiên: {loaded_edge_indices[0].shape if hasattr(loaded_edge_indices[0], 'shape') else type(loaded_edge_indices[0])}")
            # In ra giá trị của edge_indices đầu tiên nếu nó có shape (2,) để kiểm tra
            if hasattr(loaded_edge_indices[0], 'shape') and loaded_edge_indices[0].shape == (2,):
                print(f"    Giá trị của edge_indices đầu tiên (shape (2,)): {loaded_edge_indices[0]}")
        else:
            print("Không có dữ liệu nào được tải.")


if __name__=='__main__':
    main()
