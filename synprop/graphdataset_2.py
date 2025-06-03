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
    
def add_vectors(a, b):

    if len(a) != len(b):
        raise ValueError("Hai vectơ phải có cùng chiều dài.")

    result = []
    for i in range(len(a)):
        result.append(a[i] - b[i])
    return result


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
            atom_fea1=one_hot(pt.GetAtomicNumber(graph.nodes(data=True)[i]['element']),len(atom_list))


            #charge
            if np.abs(atom_data['typesGH'][0][3]) <3:
                charge_1_=atom_data['typesGH'][0][3]
                charge_1=one_hot(charge_list.index(charge_1_),len(charge_list))
            else:
                charge_1=one_hot(5,len(charge_list))

            if np.abs(atom_data['typesGH'][1][3]) <3:
                charge_2_=atom_data['typesGH'][1][3]
                charge_2=one_hot(charge_list.index(charge_2_),len(charge_list))
            else:
                charge_2=one_hot(5,len(charge_list))

            charge_atom = charge_1 + charge_2 #ver_3
            charge_change = charge_1 - charge_2


            #hybridization
            hybrid_1 = atom_data['typesGH'][0][4] 
            hybrid_2 = atom_data['typesGH'][1][4] 
                
            if hybrid_1 in hybridization:
                atom_hybrid_1=one_hot(hybridization.index(hybrid_1),len(hybridization))
            else:
                atom_hybrid_1=one_hot(3,len(hybridization))

            if hybrid_2 in hybridization:
                atom_hybrid_2=one_hot(hybridization.index(hybrid_2),len(hybridization))
            else:
                atom_hybrid_2=one_hot(3,len(hybridization))

            atom_hybrid = atom_hybrid_1 + atom_hybrid_2
            hybrid_change = add_vectors (atom_hybrid_1, atom_hybrid_2)
            
            
            atom_fea = atom_fea1 + charge_1 + charge_change + atom_hybrid_1 + atom_hybrid_change  
            print(atom_fea)
            atom_fea_graph.append(atom_fea)

        
        #bond_feature
        row, col, edge_feat_graph=[], [], []
        for idx, bond in enumerate(lst_edges_update): # bond là cặp (chỉ số nút u, chỉ số nút v)
            u = bond[0] # Chỉ số nút nguồn cho hướng u -> v
            v = bond[1] # Chỉ số nút nguồn cho hướng v -> u

            row += [u, v]
            col += [v, u]

            # # Thêm các đặc trưng cạnh mới
            order_0, order_1 = list(graph.edges(data=True))[idx][2]['order']
            standard_order = list(graph.edges(data=True))[idx][2]['standard_order']
            
            changes = []

            # Kiểm tra thuộc liên hợp
            con_0, con_1 = list(graph.edges(data=True))[idx][2]['conjugated']
            bond_con_0 = [1] if con_0 == True else [0]
            bond_con_1 = [1] if con_1 == True else [0]

            if order_0 == 1:
                edge_fea1 = [1,0,0,0] + bond_con_0
            elif order_0 == 2:
                edge_fea1 = [0,1,0,0] + bond_con_0
            elif order_0 == 3:
                edge_fea1 = [0,0,1,0] + bond_con_0
            elif order_0 == 1.5:   
                edge_fea1 = [0,0,0,1] + bond_con_0
            else:
                edge_fea1 = [0,0,0,0,0]
            
            if order_1 == 1:
                edge_fea2 = [1,0,0,0] + bond_con_1
            elif order_1 == 2:
                edge_fea2 = [0,1,0,0] + bond_con_1
            elif order_1 == 3:
                edge_fea2 = [0,0,1,0] + bond_con_1
            elif order_1 == 1.5:  
                edge_fea2 = [0,0,0,1] + bond_con_1
            else:
                edge_fea2 = [0,0,0,0,0]
            
            changes = add_vectors (edge_fea1, edge_fea2) #signma changes, pi changes, conjugated changes
            # print (changes)

            # if standard_order == 0 and order_0 == order_1: #unchaged
            #     edge_fea3 = edge_fea1 + changes[:2]
            # elif standard_order > 0 or standard_order < 0: 
            #     edge_fea3 = edge_fea1 + changes[:2] if order_0 > order_1 else edge_fea2 + changes[:2]
            # else: edge_fea3 = [0,0,0,0,0]

            if standard_order == 0 and order_0 == order_1: #unchaged
                edge_fea3 = edge_fea1 
            elif standard_order > 0 or standard_order < 0: 
                edge_fea3 = edge_fea1 if order_0 > order_1 else edge_fea2
            else: edge_fea3 = [0,0,0]


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
            
            # print(edge_fea3)
            edge_fea = edge_fea1 + edge_fea2 #+ [standard_order] #+ edge_fea5 #edge_fea3 + [standard_order]
            
            # --- THAY ĐỔI CHÍNH Ở ĐÂY ---

            # Lấy đặc trưng của hai nguyên tử tham gia liên kết
            atom_feat_u = atom_fea_graph[u] 
            atom_feat_v = atom_fea_graph[v]

            # Tạo đặc trưng có hướng cho u -> v: Ghép đặc trưng nguyên tử nguồn u với đặc trưng liên kết
            directed_feature_uv = np.concatenate((atom_feat_u, edge_fea)).tolist() # Ví dụ dùng numpy concatenate
            edge_feat_graph.append(directed_feature_uv)

            # Tạo đặc trưng có hướng cho v -> u: Ghép đặc trưng nguyên tử nguồn v với đặc trưng liên kết
            directed_feature_vu = np.concatenate((atom_feat_v, edge_fea)).tolist() # Ví dụ dùng numpy concatenate
            edge_feat_graph.append(directed_feature_vu)


        edge_index=torch.tensor([row,col])
        edge_attr=torch.tensor(np.array(edge_feat_graph),dtype=torch.float)
        node_attr=torch.tensor(np.array(atom_fea_graph),dtype=torch.float)
        y=torch.tensor(label,dtype=torch.float)
        data= Data(x=node_attr,y=y,edge_index=edge_index,edge_attr=edge_attr) ##thử bỏ ',edge_attr=edge_attr'

        return data

    def __len__(self):
        return len(self.graph)

def main():
    
    data_path='./Data/regression/e2sn2/e2sn2.csv'
    graph_path='./Data/regression/e2sn2/its_new/e2sn2.pkl.gz'
    target='ea'
    graphdata=ReactionDataset(data_path,graph_path,target)
    print(graphdata.__getitem__(8))


if __name__=='__main__':
    main()



