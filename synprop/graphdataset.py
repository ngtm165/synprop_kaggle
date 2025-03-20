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

    return [s, p_num, d_num, f_num]

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
    def __init__(self,data_path,graph_path,target):
        super(Dataset,self).__init__()
        self.graph, self.labels = read_data(data_path,graph_path,target)

    def __getitem__(self, index):
        graph=self.graph[index]
        lst_nodes=list(graph.nodes())
        lst_nodes_update=[lst_nodes.index(i) for i in lst_nodes]
        lst_edges=list(graph.edges())
        lst_edges_update=[(lst_nodes.index(u),lst_nodes.index(v)) for u,v in lst_edges]
        label=self.labels[index]

        #atom_feature
        pt = Chem.GetPeriodicTable()
        # atom_fea1=[one_hot(pt.GetAtomicNumber(graph.nodes(data=True)[i]['element']),len(atom_list))for i in lst_nodes]
        

        atom_fea_graph=[]
        for i in lst_nodes:
            atom_data = graph.nodes(data=True)[i] #ver 7_mới
            atom_fea1 = one_hot(pt.GetAtomicNumber(atom_data.get('element',0)), len(atom_list)) # Handle missing 'element'     
            
            try:
                charge = atom_data['charge']
                if abs(charge) < 3:
                    atom_fea2 = [charge]  # Lấy charge dưới dạng list
                else:
                    atom_fea2 = one_hot(5, len(charge_list))  # 'Other' charge (one-hot)
            except (KeyError, ValueError):  # Xử lý KeyError và ValueError
                atom_fea2 = one_hot(5, len(charge_list))  # Default 'other' (one-hot)
                        
            hybridization_val = atom_data.get('hybridization')
            if hybridization_val in hybridization:
                atom_fea3 = hybridization_to_spdf(hybridization_val)
            else:
                atom_fea3 = [1, 0, 0, 0] # Giá trị mặc định nếu không có hybridization_val (ví dụ: sp0)
            
            atom_fea=atom_fea1+atom_fea2+atom_fea3
            atom_fea_graph.append(atom_fea)

        
        #bond_feature
        row, col, edge_feat_graph=[], [], []
        for idx, bond in enumerate(lst_edges_update):
            row+=[bond[0],bond[1]]
            col+=[bond[1],bond[0]]
            
            # # Thêm các đặc trưng cạnh mới
            order_0, order_1 = list(graph.edges(data=True))[idx][2]['order']
            standard_order = list(graph.edges(data=True))[idx][2]['standard_order']
            
            changes = []

            #one hot encoding cho order và standard_order
            if order_0 == 1:
                edge_fea1 = [1,0,0]
            elif order_0 == 2:
                edge_fea1 = [1,1,0]
            elif order_0 == 3:
                edge_fea1 = [1,2,0]
            elif order_0 == 1.5:   
                edge_fea1 = [1,0.5,1]
            else:
                edge_fea1 = [0,0,0]
            
            if order_1 == 1:
                edge_fea2 = [1,0,0]
            elif order_1 == 2:
                edge_fea2 = [1,1,0]
            elif order_1 == 3:
                edge_fea2 = [1,2,0]
            elif order_1 == 1.5:  
                edge_fea2 = [1,0.5,1]
            else:
                edge_fea2 = [0,0,0]
            
            # print (edge_fea1, edge_fea2)
            changes = add_vectors (edge_fea1, edge_fea2) #signma changes, pi changes, conjugated changes
            # print (changes)

            if standard_order == 0 and order_0 == order_1: #unchaged
                edge_fea3 = edge_fea1 + changes[:2]
            elif standard_order > 0 or standard_order < 0: 
                edge_fea3 = edge_fea1 + changes[:2] if order_0 > order_1 else edge_fea2 + changes[:2]
            else: edge_fea3 = [0,0,0,0,0]

            # print (standard_order)
            total_standard_order = calculate_standard_order(graph, standard_order)

            # Tính toán edge_fea5 dựa trên tổng standard order
            if total_standard_order == 0:
                edge_fea5 = [0]
            elif total_standard_order == 1:
                edge_fea5 = [1]
            elif total_standard_order == -1:
                edge_fea5 = [-1]
            else:
                edge_fea5 = [total_standard_order] # handle other cases

            #Kiểm tra thuộc aromatic
            aromatic_val = graph.nodes(data=True)[list(graph.edges(data=True))[idx][0]].get('aromatic', True)
            aromatic_onehot = [1] if aromatic_val else [0]
            if aromatic_onehot == [1] and order_0 == 1.5 and order_1 == 1.5:
                edge_aromatic = [1]
            else:
                edge_aromatic = [0]
                
            edge_fea = edge_fea1 + edge_fea2 + edge_fea3 + edge_fea5 + [standard_order] + edge_aromatic #+ changes + [degree_u, degree_v, common_neighbors, order_ratio]

            edge_feat_graph.append(edge_fea)
            edge_feat_graph.append(edge_fea)

        edge_index=torch.tensor([row,col])
        edge_attr=torch.tensor(np.array(edge_feat_graph),dtype=torch.float)
        node_attr=torch.tensor(np.array(atom_fea_graph),dtype=torch.float)
        y=torch.tensor(label,dtype=torch.float)
        data= Data(x=node_attr,y=y,edge_index=edge_index,edge_attr=edge_attr)

        # print(edge_index)
        # print(edge_attr.shape)


        return data

    def __len__(self):
        return len(self.graph)

def main():
    
    data_path='./Data/regression/phosphatase/phosphatase.csv'
    graph_path='./Data/regression/phosphatase/its_origin/phosphatase.pkl.gz'
    target='Conversion'
    graphdata=ReactionDataset(data_path,graph_path,target)
    print(graphdata.__getitem__(11))

if __name__=='__main__':
    main()
