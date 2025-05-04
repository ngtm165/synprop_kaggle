import os
import json
import torch
import torch.nn as nn
from torch.optim import Adam
import random

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import optuna

root_dir=Path(__file__).resolve().parents[1]
sys.path.append(str(root_dir))
os.chdir(str(root_dir))
from synprop.model import model, train, inference
from synprop.data_wrapper_7 import data_wrapper_7 

def finetune_with_optuna(args): 
    batch_size = args.batch_size
    model_path = args.model_path + args.model_name # Đường dẫn này sẽ dùng sau khi có best_params
    monitor_path = args.monitor_folder + args.monitor_name 
    # epochs = args.epochs # Số epochs này sẽ dùng cho lần train cuối cùng
    data_path = args.data_path
    graph_path = args.graph_path
    target = args.y_column
    # Lấy các tham số Optuna từ args (cần thêm vào main_finetune.py)
    n_trials = getattr(args, 'n_trials', 50) # Dùng getattr với giá trị mặc định
    n_epochs_per_trial = getattr(args, 'n_epochs_per_trial', 15) # Dùng getattr


    data = pd.read_csv(data_path)
    out_dim = 1
    device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print("device is\t", device)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # --- Chuẩn bị Data Loaders ---

    # Chia dữ liệu (ví dụ tỷ lệ 80-10-10) - cần điều chỉnh num_workers, split_ratio nếu cần
    data_loader = data_wrapper_7(data_path, graph_path, target, batch_size, num_workers=4, valid_size=0.1, test_size=0.1)
    train_loader, val_loader, test_loader = data_loader.get_data_loaders()

    node_attr=train_loader.dataset[0].x.shape[1]
    edge_attr=train_loader.dataset[0].edge_attr.shape[1]

    print("--- model_path:", model_path)

    # --- Định nghĩa Hàm Objective cho Optuna ---
    def objective(trial):
        # 1. Đề xuất siêu tham số
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        depth = trial.suggest_int("depth", 2, 6) # Số lớp / bước lặp T
        node_hid = trial.suggest_categorical("node_hid_feats", [128, 256, 300, 512])
        # edge_hid = trial.suggest_categorical("edge_hid_feats", [128, 256, 300, 512])
        dr = trial.suggest_float("dr", 0.0, 0.5, step=0.1)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
        readout_option = trial.suggest_categorical("readout_option", [True, False])
        predict_hidden = trial.suggest_int("predict_hidden_feats", 128, 1024, step=128)
        readout_f = trial.suggest_int("readout_feats", 512, 2048, step=512) if readout_option else node_hid # Kích thước readout phụ thuộc option

        # 2. Xây dựng mô hình với tham số đề xuất
        # Đảm bảo lớp 'model' đã được import hoặc định nghĩa đúng
        current_model = model(
            node_feat=node_attr,
            edge_feat=edge_attr,
            out_dim=out_dim,
            num_layer_dmpnn=depth, # Đảm bảo tên khớp định nghĩa model
            node_hid_feats=node_hid,
            # edge_hid_feats=edge_hid,
            readout_feats=readout_f,
            predict_hidden_feats=predict_hidden,
            readout_option=readout_option,
            drop_ratio=dr
        ).to(device)

        # 3. Optimizer
        optimizer = Adam(current_model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = torch.nn.MSELoss()

    # 4. Huấn luyện ngắn hạn cho trial
    for epoch in range(args.n_epochs_per_trial): # Dùng số epoch từ args
        current_model.train()
        epoch_train_loss = []
        # Bỏ tqdm ở đây để output đỡ rối khi chạy song song nhiều trial
        for batchdata in train_loader:
            batchdata = batchdata.to(device)
            try:
                optimizer.zero_grad()
                pred = current_model(batchdata)
                labels = batchdata.y.to(device)
                loss = loss_fn(pred.view_as(labels), labels)
                if torch.isnan(loss): # Kiểm tra NaN loss
                    print(f"Warning: NaN loss encountered in trial {trial.number}, epoch {epoch}. Pruning.")
                    raise optuna.exceptions.TrialPruned() # Dừng trial nếu loss là NaN
                loss.backward()
                optimizer.step()
                epoch_train_loss.append(loss.item())
            except Exception as train_e:
                print(f"Error during training in trial {trial.number}, epoch {epoch}: {train_e}")
                raise optuna.exceptions.TrialPruned() # Dừng trial nếu có lỗi

        avg_epoch_loss = np.mean(epoch_train_loss) if epoch_train_loss else float('inf')
        print(f"Trial {trial.number} Epoch {epoch+1}/{args.n_epochs_per_trial} Train Loss: {avg_epoch_loss:.4f}")


        # --- Đánh giá trên validation và Pruning (Optional but recommended) ---
        current_model.eval()
        try:
             # Gọi hàm inference đã import để lấy validation loss
             val_rmse_p, val_mae_p, val_loss_p = inference(args, current_model, val_loader, device, loss_fn)
             if np.isnan(val_loss_p): # Kiểm tra NaN loss
                 print(f"Warning: NaN validation loss encountered in trial {trial.number}, epoch {epoch}. Pruning.")
                 raise optuna.exceptions.TrialPruned()
        except Exception as eval_e:
             print(f"Error during validation in trial {trial.number}, epoch {epoch}: {eval_e}")
             raise optuna.exceptions.TrialPruned() # Dừng trial nếu có lỗi

        trial.report(val_loss_p, epoch) # Báo cáo kết quả validation loss cho Optuna

        # Kiểm tra xem có nên dừng sớm trial này không
        if trial.should_prune():
            print(f"Trial {trial.number} pruned at epoch {epoch+1}")
            raise optuna.exceptions.TrialPruned()

    # 5. Trả về metric cuối cùng của trial (sau khi hoàn thành các epoch hoặc không bị prune)
    # Metric này đã được tính ở epoch cuối cùng trong vòng lặp trên
    print(f"Trial {trial.number} finished. Final Val Loss reported: {val_loss_p:.4f}")
    return val_loss_p # Trả về validation loss cuối cùng


# --- Hàm Main để chạy Optuna ---
def run_optimization(args):
    """Thiết lập và chạy Optuna study."""

    # --- Setup Device và Seed ---
    device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print("Device for Optuna:\t", device)
    # Set seed (Quan trọng cho việc tái lập kết quả Optuna ở mức độ nào đó)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        # Thêm các cài đặt khác để tăng tính tái lập nếu cần
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    # --- Chuẩn bị Data Loaders ---
    print("Loading data...")
    try:
        # Sử dụng data_wrapper_7 đã import
        data_loader = data_wrapper_7(args.data_path, args.graph_path, args.y_column,
                                     args.batch_size, num_workers=4, valid_size=0.1, test_size=0.1)
        train_loader, val_loader, _ = data_loader.get_data_loaders() # Chỉ cần train/val
    except Exception as e:
        print(f"Error initializing data loader: {e}")
        sys.exit(1)

    # Lấy kích thước features
    try:
        node_attr = train_loader.dataset.num_node_features
        edge_attr = train_loader.dataset.num_edge_features
        # Xác định out_dim (ví dụ cho hồi quy)
        out_dim = 1
        print(f"Node features: {node_attr}, Edge features: {edge_attr}, Out dim: {out_dim}")
    except Exception as e:
        print(f"Lỗi khi lấy feature dimensions: {e}")
        sys.exit(1)

    # --- Tạo và Chạy Optuna Study ---
    study_name = args.study_name # Lấy tên study từ args
    storage_name = f"sqlite:///{args.db_path}{study_name}.db" # Lưu trữ vào SQLite DB
    print(f"Starting Optuna study: {study_name}")
    print(f"Database storage: {storage_name}")
    Path(args.db_path).mkdir(parents=True, exist_ok=True) # Tạo thư mục nếu chưa có

    # Tạo study, lưu vào DB để có thể resume hoặc xem lại
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True, # Tải lại study nếu tên đã tồn tại
        direction="minimize" # Mục tiêu là giảm validation loss
    )

    # Đóng gói các tham số cố định cho hàm objective bằng lambda hoặc partial
    # Lambda dễ hơn trong trường hợp này
    objective_func = lambda trial: objective(trial, args, node_attr, edge_attr, out_dim, train_loader, val_loader, device)

    print(f"Running Optuna optimization for {args.n_trials} trials...")
    try:
        study.optimize(objective_func, n_trials=args.n_trials, timeout=args.timeout) # Thêm timeout nếu muốn giới hạn thời gian tổng
    except KeyboardInterrupt:
         print("Optimization stopped by user.")
    except Exception as opt_e:
         print(f"An error occurred during optimization: {opt_e}")


    # --- In và Lưu kết quả tốt nhất ---
    print("\n" + "="*30 + " Optuna Optimization Finished " + "="*30)
    try:
        print(f"Study '{study.study_name}' finished with {len(study.trials)} trials.")
        best_trial = study.best_trial
        best_params = best_trial.params
        best_value = best_trial.value
        print(f"Best validation loss found: {best_value:.4f}")
        print("Best hyperparameters found:")
        print(json.dumps(best_params, indent=4))

        # Lưu best_params vào file json
        best_params_path = os.path.join(args.output_dir, f"{study_name}_best_params.json")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        with open(best_params_path, "w") as f:
            json.dump(best_params, f, indent=4)
        print(f"Best parameters saved to {best_params_path}")

    except optuna.exceptions.OptunaError:
         print(f"Optuna study '{study.study_name}' finished without finding any successful trials.")
    except Exception as e:
         print(f"An unexpected error occurred retrieving results: {e}")


# --- Parser cho Command Line Arguments ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Optuna Hyperparameter Optimization for GNN")

    # Tham số dữ liệu & mô hình cơ bản (giống main_finetune)
    parser.add_argument("--data_path", type=str, default='./Data/regression/lograte/lograte.csv')
    parser.add_argument("--graph_path", type=str, default='./Data/regression/lograte/its_new/lograte.pkl.gz')
    parser.add_argument("--y_column", type=str, default="lograte")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=27407)

    # Tham số cho Optuna
    parser.add_argument("--n_trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--n_epochs_per_trial", type=int, default=15, help="Number of epochs per Optuna trial")
    parser.add_argument("--study_name", type=str, default="gnn_optimization", help="Name for the Optuna study")
    parser.add_argument("--db_path", type=str, default="./optuna_db/", help="Directory to save Optuna study database")
    parser.add_argument("--output_dir", type=str, default="./optuna_results/", help="Directory to save best parameters JSON")
    parser.add_argument("--timeout", type=int, default=None, help="Optional timeout for Optuna study in seconds")

    # Tham số cho không gian tìm kiếm của Optuna (ví dụ)
    # Bạn có thể đặt các khoảng giá trị ở đây thay vì hardcode trong objective
    parser.add_argument('--min_depth', type=int, default=2)
    parser.add_argument('--max_depth', type=int, default=6)
    parser.add_argument('--node_hid_choices', type=int, nargs='+', default=[128, 256, 300, 512])
    parser.add_argument('--edge_hid_choices', type=int, nargs='+', default=[128, 256, 300, 512])
    parser.add_argument('--min_dr', type=float, default=0.0)
    parser.add_argument('--max_dr', type=float, default=0.5)
    # Thêm các args khác nếu cần thiết cho hàm inference hoặc các phần khác

    args = parser.parse_args()

    # Chạy hàm tối ưu hóa
    run_optimization(args)