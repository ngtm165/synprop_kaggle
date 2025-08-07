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

def one_hot(idx, length):
    lst_onehot = [0 for i in range(length)]
    lst_onehot[idx] = 1
    return lst_onehot

def neighbors_to_quantum_numbers(neighbors):
    element_to_atomic_number_1 = {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
        'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
        'K': 19, 'Ca': 20, 'Br': 35
    }
    if not neighbors:  # Kiểm tra nếu danh sách rỗng
        return [0]
    neighbor_atomic_numbers = []
    for element in neighbors:
        if element not in element_to_atomic_number_1:
            neighbor_atomic_numbers.append(0)  # Thêm 0 nếu không tìm thấy
        else:
            neighbor_atomic_numbers.append(element_to_atomic_number_1[element])
    return neighbor_atomic_numbers

def element_to_quantum_numbers(element):
    element_to_atomic_number = {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
        'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
        'K': 19, 'Ca': 20, 'Br': 35
    }
    if element not in element_to_atomic_number:
        return None
    atomic_number = element_to_atomic_number[element]
    return atomic_number_to_quantum_numbers(atomic_number)

def atomic_number_to_quantum_numbers(atomic_number):
    electron_configuration = get_electron_configuration(atomic_number)
    outer_subshell = electron_configuration[-1]
    # print(atomic_number)
    n = int(outer_subshell[0])
    l = 0 if outer_subshell[1] == 's' else 1 if outer_subshell[1] == 'p' else 2 if outer_subshell[1] == 'd' else 3

    num_orbitals = 2 * l + 1
    num_electrons = int(outer_subshell[2:]) # Cần tính số electron trong phân lớp ngoài cùng

    orbitals = [0] * num_orbitals
    spin = 1
    # last_orbital_index = 0  # Theo dõi vị trí electron cuối cùng
    last_spin = 1 # Thêm biến để theo dõi spin electron cuối cùng


    for i in range(num_electrons):
        orbital_index = i % num_orbitals
        if orbitals[orbital_index] == 0:
            orbitals[orbital_index] = spin
            last_spin = spin # Cập nhật spin electron cuối cùng
        else:
            orbitals[orbital_index] = 2
            last_spin = -spin # Cập nhật spin electron cuối cùng
        # spin *= -1

    # Ánh xạ orbital_index sang ml
    ml_map = list(range(-l, l + 1))  # Tạo danh sách [-l, -l+1, ..., l-1, l]
    ml = ml_map[orbital_index]

    # Xác định ms dựa trên spin cuối cùng
    ms = 0.5 if last_spin == 1 else -0.5
    
    empty_orbitals = []
    single_electron_orbitals = []
    full_orbitals = []

    for i, orbital_state in enumerate(orbitals):
        if orbital_state == 0:
            empty_orbitals.append(ml_map[i])
        elif orbital_state == 1 or orbital_state == -1 :
            single_electron_orbitals.append(ml_map[i])
            

    for i, orbital_state in enumerate(orbitals):
        if orbital_state == 2:
            full_orbitals.append(ml_map[i])

    
    # Tính tổng số electron lớp ngoài cùng
    outer_electrons = 0
    for subshell in electron_configuration:
        if int(subshell[0]) == n:  # Kiểm tra nếu phân lớp thuộc lớp ngoài cùng
            outer_electrons += int(subshell[-1])
    
    # Tính tổng số orbital lớp ngoài cùng
    outer_orbitals = 0

    max_l = []
    if n >= 1:
        for i in range(min(n, 4)):  # Chỉ lấy tối đa 4 giá trị của l
            max_l.append(i)
            outer_orbitals += 2 * i + 1
            
    # xac dinh hoa tri
    e = outer_orbitals if outer_electrons > outer_orbitals else outer_electrons
        #Thêm logic xử lý riêng cho atomic_number 8 và 9 nếu cần.
    if atomic_number == 8 or atomic_number == 9:
        e = len(single_electron_orbitals)
    
    return (n, l, ml, ms, e)


def get_electron_configuration(atomic_number):
    subshells = ['1s', '2s', '2p', '3s', '3p', '4s', '3d', '4p', '5s', '4d', '5p', '6s', '4f', '5d', '6p', '7s', '5f', '6d', '7p']
    max_electrons = [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 14, 10, 6]
    electron_configuration = []
    remaining_electrons = atomic_number
    for i in range(len(subshells)):
        if remaining_electrons <= 0:
            break
        if remaining_electrons <= max_electrons[i]:
            electron_configuration.append(subshells[i] + str(remaining_electrons))
            remaining_electrons = 0
        else:
            electron_configuration.append(subshells[i] + str(max_electrons[i]))
            remaining_electrons -= max_electrons[i]
    return electron_configuration

def hybridization_to_spdf(hybridization):
    hybridization = hybridization.lower()

    s = hybridization.count('s')
    p = hybridization.count('p')
    d = hybridization.count('d')
    f = hybridization.count('f')

    p_num = 0
    d_num = 0
    f_num = 0

    if 'p' in hybridization:
        p_index = hybridization.find('p')
        if p_index + 1 < len(hybridization) and hybridization[p_index + 1].isdigit():
            num_str = ''
            for char in hybridization[p_index + 1:]:
                if char.isdigit():
                    num_str += char
                else:
                    break
            if num_str:
                p_num = int(num_str)
            else:
                p_num = 1

    if 'd' in hybridization:
        d_index = hybridization.find('d')
        if d_index + 1 < len(hybridization) and hybridization[d_index + 1].isdigit():
            num_str = ''
            for char in hybridization[d_index + 1:]:
                if char.isdigit():
                    num_str += char
                else:
                    break
            if num_str:
                d_num = int(num_str)
            else:
                d_num = 1

    if 'f' in hybridization:
        f_index = hybridization.find('f')
        if f_index + 1 < len(hybridization) and hybridization[f_index + 1].isdigit():
            num_str = ''
            for char in hybridization[f_index + 1:]:
                if char.isdigit():
                    num_str += char
                else:
                    break
            if num_str:
                f_num = int(num_str)
            else:
                f_num = 1

    total = s + p_num + d_num + f_num #thử bỏ f_num
    
    return [s, p_num, d_num, f_num], total
    
    # if total == 0:
    #   return [0,0,0,0]

    # return [s / total, p_num / total, d_num / total, f_num / total], total

def lone_pairs (total, sigma):
    lone = total - sigma 
    return lone 

def add_vectors(a, b):

    if len(a) != len(b):
        raise ValueError("Hai vectơ phải có cùng chiều dài.")

    result = []
    for i in range(len(a)):
        result.append(a[i] - b[i])
    return result

def calculate_standard_order(graph, standard_order):
    """Tính tổng standard order từ thông tin đồ thị."""
    standard_orders = []
    for u, v, data in graph.edges(data=True):
        standard_orders.append(data['standard_order'])
    return sum(standard_orders)


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
        

            #hybridization
            hybrid_1 = atom_data['typesGH'][0][4] 
            hybrid_2 = atom_data['typesGH'][1][4] 

            atom_hybrid_1, total_1 = hybridization_to_spdf(hybrid_1)
            atom_hybrid_2, total_2 = hybridization_to_spdf(hybrid_2)

            atom_hybrid = atom_hybrid_1 + atom_hybrid_2
            hybrid_change = add_vectors (atom_hybrid_1, atom_hybrid_2)

            # Tích hợp số lượng tử
            element = graph.nodes(data=True)[i]['element']
            quantum_numbers = element_to_quantum_numbers(element)
            if quantum_numbers:
                n, l, ml, ms, e = quantum_numbers
                quantum_features = [n, l, ml, ms]  # Chuyển thành list
            else:
                quantum_features = [0, 0, 0, 0]  # Giá trị mặc định nếu không tìm thấy


            # Liên kết tối đa (valence electrons)
            e_max = [e]
            
            # Mã hóa one-hot cho các thuộc tính bổ sung
            hcount_1 = atom_data['typesGH'][0][2]
            hcount_2 = atom_data['typesGH'][1][2]
        
            # # Featurize số lượng nguyên tố neighbors

            neighbor_1 = atom_data['typesGH'][0][5] 
            neighbor_2 = atom_data['typesGH'][1][5] 

            neighbor_count_1 = len(neighbor_1)
            neighbor_count_2 = len(neighbor_2)
            neighbor_change = neighbor_count_1 - neighbor_count_2

            neighbor_elements_1 = neighbors_to_quantum_numbers(neighbor_1)
            neighbor_elements_2 = neighbors_to_quantum_numbers(neighbor_2)

            #Kiểm tra thuyết
            h_val_1 = neighbor_elements_1.count(1)
            h_val_2 = neighbor_elements_2.count(1)
            h_1 = [hcount_1 + h_val_1] 
            h_2 = [hcount_2 + h_val_2] 
            hcount = [hcount_1 + h_val_1] + [hcount_2 + h_val_2] 

            hcount_change = add_vectors (h_1, h_2)

            if h_val_1 == 0:
                sigma_1 = hcount_1 + h_val_1 + neighbor_count_1
            else: sigma_1 = neighbor_count_1

            if h_val_2 == 0:
                sigma_2 = hcount_2 + h_val_2 + neighbor_count_2   
            else: sigma_2 = neighbor_count_2         

            lone_1 = lone_pairs (total_1, sigma_1)
            lone_2 = lone_pairs (total_2, sigma_2)

            atom_hybrid_p_1 = atom_hybrid_1 + [sigma_1] + [lone_1]
            atom_hybrid_p_2 = atom_hybrid_2 + [sigma_2] + [lone_2]
            atom_hybrid_p = atom_hybrid_p_1 + atom_hybrid_p_2
            atom_hybrid_change = add_vectors (atom_hybrid_p_1, atom_hybrid_p_2)
            
            atom_fea = quantum_features + e_max + [charge_1] + [charge_change] + h_1 + hcount_change + atom_hybrid_p_1 + atom_hybrid_change + [neighbor_count_1] + [neighbor_change] #5.1

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

            order_0, order_1 = list(graph.edges(data=True))[idx][2]['order']
            standard_order = list(graph.edges(data=True))[idx][2]['standard_order']
            
            changes = []

            con_0, con_1 = list(graph.edges(data=True))[idx][2]['conjugated']
            bond_con_0 = [1.0] if con_0 == True else [0.0]
            bond_con_1 = [1.0] if con_1 == True else [0.0]

            if order_0 == 1:
                edge_fea1 = [1.0,0.0] + bond_con_0
            elif order_0 == 2:
                edge_fea1 = [1.0,1.0] + bond_con_0
            elif order_0 == 3:
                edge_fea1 = [1.0,2.0] + bond_con_0
            elif order_0 == 1.5:   
                edge_fea1 = [1.0,0.5] + bond_con_0
            else:
                edge_fea1 = [0.0,0.0,0.0]
            
            if order_1 == 1:
                edge_fea2 = [1.0,0.0] + bond_con_1
            elif order_1 == 2:
                edge_fea2 = [1.0,1.0] + bond_con_1
            elif order_1 == 3:
                edge_fea2 = [1.0,2.0]+ bond_con_1
            elif order_1 == 1.5:  
                edge_fea2 = [1.0,0.5] + bond_con_1
            else:
                edge_fea2 = [0.0,0.0,0.0]
            
            changes = add_vectors (edge_fea1, edge_fea2) #signma changes, pi changes, conjugated changes
            
            # print(edge_fea3)
            edge_fea = edge_fea1 + edge_fea2 

            edge_feat_graph.append(edge_fea)
            edge_feat_graph.append(edge_fea)

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
        row_RC, col_RC = [], []
        
        num_original_atoms = len(lst_nodes_update)

        unique_rc_nodes = set()
        for bond_data in reaction_center_data:
            unique_rc_nodes.update(bond_data['node_features'].keys())
            
        sorted_rc_nodes = sorted(list(unique_rc_nodes))

        node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_rc_nodes, start=0)}
        
        reindexed_rc_data = []
        for bond_data in reaction_center_data:
            # Lấy ra các chỉ số cũ
            u_old, v_old = bond_data['node_features'].keys()
            
            # Tra cứu các chỉ số mới từ bản đồ
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
            edge_feat_graph_RC.append(RC_edge_features) 
            edge_feat_graph_RC.append(RC_edge_features) 
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

        # BƯỚC 3: THÊM NÚT VÀO DANH SÁCH
        atom_fea_graph_super.append(supernode_node_feature)

        # BƯỚC 4: THÊM CÁC CẠNH MỚI
        # print("\nThêm các cạnh nối siêu nút với RC:")
        for rc_node_old_index in (list(unique_rc_nodes)): # Sắp xếp để in ra cho đẹp
            rc_node_new_index = node_mapping[rc_node_old_index]
            
            row_super.append(supernode_index)
            col_super.append(rc_node_new_index)
            edge_feat_graph_super.append(supernode_edge_feature)
            
            row_super.append(rc_node_new_index)
            col_super.append(supernode_index)
            edge_feat_graph_super.append(supernode_edge_feature)

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

    def __len__(self):
        return len(self.graph)


def save_to_npz(data_dict, output_path):
    """
    Hàm trợ giúp để lưu một tập hợp các thuộc tính đồ thị vào file .npz.
    Phiên bản này đã sửa lỗi ValueError broadcast.

    Args:
        data_dict (dict): Từ điển chứa các danh sách numpy arrays.
        output_path (str): Đường dẫn đến file .npz đầu ra.
    """
    print(f"\nĐang chuẩn bị và lưu dữ liệu vào file: {output_path}")

    # Danh sách các thuộc tính cần lưu
    attributes = ['node_attrs', 'edge_indices', 'edge_attrs', 'ys']
    data_to_save = {}

    for attr in attributes:
        # Lấy danh sách các mảng numpy cho thuộc tính hiện tại
        list_of_arrays = data_dict[attr]
        num_samples = len(list_of_arrays)
        
        # Tạo một mảng rỗng với dtype=object có kích thước phù hợp
        object_array = np.empty(num_samples, dtype=object)
        
        # Điền từng mảng con vào mảng object
        for i in range(num_samples):
            object_array[i] = list_of_arrays[i]
            
        data_to_save[attr] = object_array

    # Lưu tất cả các mảng object đã được tạo
    np.savez_compressed(
        output_path,
        node_attrs=data_to_save['node_attrs'],
        edge_indices=data_to_save['edge_indices'],
        edge_attrs=data_to_save['edge_attrs'],
        ys=data_to_save['ys']
    )
    print(f"✅ Đã lưu thành công {len(data_to_save['node_attrs'])} mẫu vào '{output_path}'.")

def main():
    """
    Hàm chính để xử lý các tập dữ liệu (train, val, test) và lưu các đặc trưng
    của đồ thị Normal, RC, và Supernode vào các file .npz riêng biệt.
    """
    folder_list = ['test', 'val', 'train']
    base_data_path = './Data/regression/barriers_cycloadd'
    output_folder = './output/RC/nhap1/barriers_cycloadd'
    os.makedirs(output_folder, exist_ok=True) 

    for dataset_type in folder_list:
        print(f"\n{'='*20} BẮT ĐẦU XỬ LÝ: {dataset_type.upper()} {'='*20}")
        data_path = os.path.join(base_data_path, f'{dataset_type}.csv')
        graph_path = os.path.join(base_data_path, 'its_new', f'barriers_cycloadd_aam_{dataset_type}.pkl.gz')
        target = 'G_act'
        
        # 1. Khởi tạo Dataset
        graphdata = ReactionDataset(data_path, graph_path, target)
        print(graphdata.__getitem__(8))

        # 2. Chuẩn bị cấu trúc để lưu trữ dữ liệu
        # Sử dụng từ điển để quản lý dữ liệu cho từng loại đồ thị
        all_data = {
            'normal': {'node_attrs': [], 'edge_indices': [], 'edge_attrs': [], 'ys': []},
            'rc': {'node_attrs': [], 'edge_indices': [], 'edge_attrs': [], 'ys': []},
            'super': {'node_attrs': [], 'edge_indices': [], 'edge_attrs': [], 'ys': []}
        }
        
        # 3. Trích xuất dữ liệu từ từng mẫu
        print(f"Đang trích xuất dữ liệu từ {len(graphdata)} mẫu trong '{dataset_type}'...")
        for i in range(len(graphdata)):
            try:
                # Lấy cả 3 object Data từ __getitem__
                data_normal, data_rc, data_super = graphdata[i]
                
                # Tạo một map để dễ dàng lặp qua
                data_map = {
                    'normal': data_normal,
                    'rc': data_rc,
                    'super': data_super
                }

                # Lặp qua từng loại đồ thị và lưu trữ các thuộc tính
                for name, data_obj in data_map.items():
                    # Chỉ thêm dữ liệu nếu đồ thị không rỗng
                    if data_obj.num_nodes > 0:
                        all_data[name]['node_attrs'].append(data_obj.x.numpy())
                        all_data[name]['edge_indices'].append(data_obj.edge_index.numpy())
                        all_data[name]['edge_attrs'].append(data_obj.edge_attr.numpy())
                        all_data[name]['ys'].append(data_obj.y.numpy())

                if (i + 1) % 100 == 0: # In tiến độ mỗi 100 mẫu
                    print(f"  Đã xử lý {i+1}/{len(graphdata)} mẫu.")

            except Exception as e:
                print(f"❌ Lỗi khi xử lý mẫu {i} trong tập {dataset_type}: {e}")
                # continue # Bỏ qua mẫu lỗi và tiếp tục

        # 4. Lưu dữ liệu đã xử lý vào các file .npz riêng biệt
        for name, data_dict in all_data.items():
            if not data_dict['node_attrs']:
                print(f"\n⚠️ Không có dữ liệu để lưu cho loại '{name}' trong tập '{dataset_type}'. Bỏ qua.")
                continue
                
            output_filename = f'barriers_cycloadd_aam_{dataset_type}_{name}_processed_data.npz'
            output_npz_file = os.path.join(output_folder, output_filename)
            save_to_npz(data_dict, output_npz_file)
            
        print(f"\n{'='*20} HOÀN TẤT XỬ LÝ: {dataset_type.upper()} {'='*20}")

if __name__ == '__main__':

    main()