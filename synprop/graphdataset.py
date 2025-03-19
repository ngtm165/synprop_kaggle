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
    lst_onehot=[0 for i in range(length)]
    lst_onehot[idx]=1
    return lst_onehot



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
            atom_fea1=one_hot(pt.GetAtomicNumber(graph.nodes(data=True)[i]['element']),len(atom_list))
            if np.abs(graph.nodes(data=True)[i]['charge']) <3:
                charge=graph.nodes(data=True)[i]['charge']
                atom_fea2=one_hot(charge_list.index(charge),len(charge_list))
            else:
                atom_fea2=one_hot(5,len(charge_list))

            if graph.nodes(data=True)[i]['hybridization'] in hybridization:
                atom_fea3=one_hot(hybridization.index(graph.nodes(data=True)[i]['hybridization']),len(hybridization))
            else:
                atom_fea3=one_hot(3,len(hybridization))
            
            # if graph.nodes(data=True)[i]['explicit_valence'] in valence:
            #     atom_fea4=one_hot(valence.index(graph.nodes(data=True)[i]['explicit_valence']),len(valence))
            # else:
            #     atom_fea4=one_hot(6,len(valence))
            atom_fea=atom_fea1+atom_fea2+atom_fea3
            atom_fea_graph.append(atom_fea)

        
        #bond_feature
        row, col, edge_feat_graph=[], [], []
        for idx,bond in enumerate(lst_edges_update):
            row+=[bond[0],bond[1]]
            col+=[bond[1],bond[0]]

            if np.max(list(graph.edges(data=True))[idx][2]['order'])==1:
                edge_fea1=one_hot(0,len(bond_type1))
            elif np.max(list(graph.edges(data=True))[idx][2]['order'])==2:
                edge_fea1=one_hot(1,len(bond_type1))
            elif np.max(list(graph.edges(data=True))[idx][2]['order'])==3:
                edge_fea1=one_hot(2,len(bond_type1))
            else:
                edge_fea1=one_hot(3,len(bond_type1))

            if list(graph.edges(data=True))[idx][2]['standard_order'] ==0:

                edge_fea2=one_hot(0,len(bond_type2))
                # edge_fea2=[x*10 for x in edge_fea2]
            elif list(graph.edges(data=True))[idx][2]['standard_order'] <0:

                edge_fea2=one_hot(1,len(bond_type2))
                # edge_fea2=[x*20 for x in edge_fea2]
            else:

                edge_fea2=one_hot(2,len(bond_type2))
                # edge_fea2=[x*20 for x in edge_fea2]
            edge_fea=edge_fea1+edge_fea2
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
    
    data_path='./Data/regression/lograte/lograte.csv'
    graph_path='./Data/regression/lograte/its_origin/lograte.pkl.gz'
    target='lograte'
    graphdata=ReactionDataset(data_path,graph_path,target)
    print(graphdata.__getitem__(0))

if __name__=='__main__':
    main()
