import os
import numpy as np
root_dir = str(Path(__file__).resolve().parents[1])
os.chdir(root_dir)
from pathlib import Path


train_npz_normal = np.load(f'../chemprop/data/RC/nhap/barriers_sn2/barriers_sn2_aam_train_normal.npz', allow_pickle=True)
train_v_normal = train_npz_normal['node_attrs']
train_e_normal = train_npz_normal['edge_attrs']
train_idx_g_normal = train_npz_normal['edge_indices']
train_y_normal = train_npz_normal['ys'] 

val_npz_normal = np.load(f'../chemprop/data/RC/nhap/barriers_sn2/barriers_sn2_aam_val_normal.npz', allow_pickle=True)
val_v_normal = val_npz_normal['node_attrs']
val_e_normal = val_npz_normal['edge_attrs']
val_idx_g_normal = val_npz_normal['edge_indices']
val_y_normal = val_npz_normal['ys'] 

test_npz_normal = np.load(f'../chemprop/data/RC/nhap/barriers_sn2/barriers_sn2_aam_test_normal.npz', allow_pickle=True)
test_v_normal = test_npz_normal['node_attrs']
test_e_normal = test_npz_normal['edge_attrs']
test_idx_g_normal = test_npz_normal['edge_indices']
test_y_normal = test_npz_normal['ys'] 

train_npz_rc = np.load(f'../chemprop/data/RC/nhap/barriers_sn2/barriers_sn2_aam_train_rc.npz', allow_pickle=True)
train_v_rc = train_npz_rc['node_attrs']
train_e_rc = train_npz_rc['edge_attrs']
train_idx_g_rc = train_npz_rc['edge_indices']
train_y_rc = train_npz_rc['ys'] 

val_npz_rc = np.load(f'../chemprop/data/RC/nhap/barriers_sn2/barriers_sn2_aam_val_rc.npz', allow_pickle=True)
val_v_rc = val_npz_rc['node_attrs']
val_e_rc = val_npz_rc['edge_attrs']
val_idx_g_rc = val_npz_rc['edge_indices']
val_y_rc = val_npz_rc['ys'] 

test_npz_rc = np.load(f'../chemprop/data/RC/nhap/barriers_sn2/barriers_sn2_aam_test_rc.npz', allow_pickle=True)
test_v_rc = test_npz_rc['node_attrs']
test_e_rc = test_npz_rc['edge_attrs']
test_idx_g_rc = test_npz_rc['edge_indices']
test_y_rc = test_npz_rc['ys'] 

train_npz_super = np.load(f'../chemprop/data/RC/nhap/barriers_sn2/barriers_sn2_aam_train_super.npz', allow_pickle=True)
train_v_super = train_npz_super['node_attrs']
train_e_super = train_npz_super['edge_attrs']
train_idx_g_super = train_npz_super['edge_indices']
train_y_super = train_npz_super['ys'] 

val_npz_super = np.load(f'../chemprop/data/RC/nhap/barriers_sn2/barriers_sn2_aam_val_super.npz', allow_pickle=True)
val_v_super = val_npz_super['node_attrs']
val_e_super = val_npz_super['edge_attrs']
val_idx_g_super = val_npz_super['edge_indices']
val_y_super = val_npz_super['ys'] 

test_npz_super = np.load(f'../chemprop/data/RC/nhap/barriers_sn2/barriers_sn2_aam_test_super.npz', allow_pickle=True)
test_v_super = test_npz_super['node_attrs']
test_e_super = test_npz_super['edge_attrs']
test_idx_g_super = test_npz_super['edge_indices']
test_y_super = test_npz_super['ys'] 

train_v = np.concatenate((train_v_normal, train_v_rc, train_v_super), axis=0)
train_e = np.concatenate((train_e_normal, train_e_rc, train_e_super), axis=0)
train_idx_g = np.concatenate((train_idx_g_normal, train_idx_g_rc, train_idx_g_super), axis=0)
train_y = np.concatenate((train_y_normal, train_y_rc, train_y_super), axis=0)

val_v = np.concatenate((val_v_normal, val_v_rc, val_v_super), axis=0)
val_e = np.concatenate((val_e_normal, val_e_rc, val_e_super), axis=0)
val_idx_g = np.concatenate((val_idx_g_normal, val_idx_g_rc, val_idx_g_super), axis=0)
val_y = np.concatenate((val_y_normal, val_y_rc, val_y_super), axis=0)

test_v = np.concatenate((test_v_normal, test_v_rc, test_v_super), axis=0)
test_e = np.concatenate((test_e_normal, test_e_rc, test_e_super), axis=0)
test_idx_g = np.concatenate((test_idx_g_normal, test_idx_g_rc, test_idx_g_super), axis=0)
test_y = np.concatenate((test_y_normal, test_y_rc, test_y_super), axis=0)