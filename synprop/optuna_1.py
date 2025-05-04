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
from synprop.model import model, train, inference # Ensure these are correctly importable
from synprop.data_wrapper_7 import data_wrapper_7 # Ensure this is correctly importable

# The finetune_with_optuna function is removed as its core logic (objective)
# is moved into run_optimization, and it wasn't being called anyway.

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
        # Using .num_node_features and .num_edge_features if available in data_wrapper_7's dataset object
        # If not, fallback to inspecting the first item (less robust)
        if hasattr(train_loader.dataset, 'num_node_features'):
             node_attr = train_loader.dataset.num_node_features
        else:
             print("Warning: dataset has no 'num_node_features'. Inferring from first item.")
             node_attr = train_loader.dataset[0].x.shape[1]

        if hasattr(train_loader.dataset, 'num_edge_features'):
             edge_attr = train_loader.dataset.num_edge_features
        else:
             print("Warning: dataset has no 'num_edge_features'. Inferring from first item.")
             edge_attr = train_loader.dataset[0].edge_attr.shape[1]

        # Xác định out_dim (ví dụ cho hồi quy)
        out_dim = 1
        print(f"Node features: {node_attr}, Edge features: {edge_attr}, Out dim: {out_dim}")
    except Exception as e:
        print(f"Lỗi khi lấy feature dimensions: {e}")
        # Try accessing the first element as a fallback if properties don't exist
        try:
            print("Attempting fallback to inspect first data item...")
            first_data = train_loader.dataset[0]
            node_attr = first_data.x.shape[1]
            edge_attr = first_data.edge_attr.shape[1]
            out_dim = 1 # Assuming regression
            print(f"(Fallback) Node features: {node_attr}, Edge features: {edge_attr}, Out dim: {out_dim}")
        except Exception as fallback_e:
            print(f"Fallback failed. Error getting feature dimensions: {fallback_e}")
            sys.exit(1)


    # --- Định nghĩa Hàm Objective cho Optuna (MOVED HERE) ---
    def objective(trial, current_args, node_attr_obj, edge_attr_obj, out_dim_obj, train_loader_obj, val_loader_obj, device_obj):
        # Using passed arguments with _obj suffix to avoid potential conflicts
        # 1. Đề xuất siêu tham số
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        depth = trial.suggest_int("depth", current_args.min_depth, current_args.max_depth) # Use args for range
        node_hid = trial.suggest_categorical("node_hid_feats", current_args.node_hid_choices) # Use args for choices
        # edge_hid = trial.suggest_categorical("edge_hid_feats", current_args.edge_hid_choices) # Uncomment if edge_hid is used in model
        dr = trial.suggest_float("dr", current_args.min_dr, current_args.max_dr, step=0.1) # Use args for range
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
        readout_option = trial.suggest_categorical("readout_option", [True, False])
        predict_hidden = trial.suggest_int("predict_hidden_feats", 128, 1024, step=128)
        # Kích thước readout phụ thuộc option
        readout_f = trial.suggest_int("readout_feats", 512, 2048, step=512) if readout_option else node_hid

        # 2. Xây dựng mô hình với tham số đề xuất
        # Đảm bảo lớp 'model' đã được import hoặc định nghĩa đúng
        current_model = model(
            node_feat=node_attr_obj,
            edge_feat=edge_attr_obj,
            out_dim=out_dim_obj,
            num_layer_dmpnn=depth, # Đảm bảo tên khớp định nghĩa model
            node_hid_feats=node_hid,
            # edge_hid_feats=edge_hid, # Uncomment if used
            readout_feats=readout_f,
            predict_hidden_feats=predict_hidden,
            readout_option=readout_option,
            drop_ratio=dr
        ).to(device_obj)

        # 3. Optimizer
        optimizer = Adam(current_model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = torch.nn.MSELoss()

        # 4. Huấn luyện ngắn hạn cho trial
        n_epochs_per_trial = getattr(current_args, 'n_epochs_per_trial', 15) # Get epochs per trial
        val_loss_p = float('inf') # Initialize validation loss

        for epoch in range(n_epochs_per_trial): # Dùng số epoch từ args
            current_model.train()
            epoch_train_loss = []
            # Bỏ tqdm ở đây để output đỡ rối khi chạy song song nhiều trial
            for batchdata in train_loader_obj:
                batchdata = batchdata.to(device_obj)
                try:
                    optimizer.zero_grad()
                    pred = current_model(batchdata)
                    labels = batchdata.y.to(device_obj)
                    loss = loss_fn(pred.view_as(labels), labels)
                    if torch.isnan(loss): # Kiểm tra NaN loss
                        print(f"Warning: NaN loss encountered in trial {trial.number}, epoch {epoch+1}. Pruning.")
                        raise optuna.exceptions.TrialPruned() # Dừng trial nếu loss là NaN
                    loss.backward()
                    optimizer.step()
                    epoch_train_loss.append(loss.item())
                except RuntimeError as e: # Catch specific runtime errors like OOM
                    if "out of memory" in str(e).lower():
                        print(f"Warning: CUDA out of memory in trial {trial.number}, epoch {epoch+1}. Pruning.")
                        torch.cuda.empty_cache() # Try to free memory
                        raise optuna.exceptions.TrialPruned()
                    else:
                        print(f"Runtime error during training in trial {trial.number}, epoch {epoch+1}: {e}")
                        raise optuna.exceptions.TrialPruned() # Prune for other runtime errors too
                except Exception as train_e:
                    print(f"Error during training in trial {trial.number}, epoch {epoch+1}: {train_e}")
                    raise optuna.exceptions.TrialPruned() # Dừng trial nếu có lỗi khác

            avg_epoch_loss = np.mean(epoch_train_loss) if epoch_train_loss else float('inf')
            print(f"Trial {trial.number} Epoch {epoch+1}/{n_epochs_per_trial} Train Loss: {avg_epoch_loss:.4f}")

            # --- Đánh giá trên validation và Pruning (Optional but recommended) ---
            current_model.eval()
            try:
                # Gọi hàm inference đã import để lấy validation loss
                # Pass current_args instead of the global args if inference needs specific args
                # Note: Ensure the 'inference' function signature matches how it's called here.
                # It might need `args`, `model`, `loader`, `device`, `loss_fn`.
                val_rmse_p, val_mae_p, val_loss_p = inference(current_args, current_model, val_loader_obj, device_obj, loss_fn)

                if np.isnan(val_loss_p): # Kiểm tra NaN loss
                    print(f"Warning: NaN validation loss encountered in trial {trial.number}, epoch {epoch+1}. Pruning.")
                    raise optuna.exceptions.TrialPruned()
            except RuntimeError as e: # Catch specific runtime errors like OOM during inference
                 if "out of memory" in str(e).lower():
                     print(f"Warning: CUDA out of memory during validation in trial {trial.number}, epoch {epoch+1}. Pruning.")
                     torch.cuda.empty_cache() # Try to free memory
                     raise optuna.exceptions.TrialPruned()
                 else:
                     print(f"Runtime error during validation in trial {trial.number}, epoch {epoch+1}: {e}")
                     raise optuna.exceptions.TrialPruned() # Prune for other runtime errors too
            except Exception as eval_e:
                 print(f"Error during validation in trial {trial.number}, epoch {epoch+1}: {eval_e}")
                 raise optuna.exceptions.TrialPruned() # Dừng trial nếu có lỗi khác

            trial.report(val_loss_p, epoch) # Báo cáo kết quả validation loss cho Optuna

            # Kiểm tra xem có nên dừng sớm trial này không
            if trial.should_prune():
                print(f"Trial {trial.number} pruned at epoch {epoch+1}")
                raise optuna.exceptions.TrialPruned()

        # 5. Trả về metric cuối cùng của trial (sau khi hoàn thành các epoch hoặc không bị prune)
        # Metric này đã được tính ở epoch cuối cùng trong vòng lặp trên
        print(f"Trial {trial.number} finished. Final Val Loss reported: {val_loss_p:.4f}")
        # Ensure val_loss_p has a valid value before returning
        if np.isinf(val_loss_p) or np.isnan(val_loss_p):
             print(f"Warning: Trial {trial.number} finished with invalid loss ({val_loss_p}). Returning large value.")
             return float('inf') # Return a large value if something went wrong

        return val_loss_p # Trả về validation loss cuối cùng


    # --- Tạo và Chạy Optuna Study ---
    study_name = args.study_name # Lấy tên study từ args
    storage_name = f"sqlite:///{args.db_path}{study_name}.db" # Lưu trữ vào SQLite DB
    print(f"Starting Optuna study: {study_name}")
    print(f"Database storage: {storage_name}")
    Path(args.db_path).mkdir(parents=True, exist_ok=True) # Tạo thư mục nếu chưa có

    # Create study, load if exists, specify direction
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True, # Tải lại study nếu tên đã tồn tại
        direction="minimize", # Mục tiêu là giảm validation loss
        pruner=optuna.pruners.MedianPruner() # Add a pruner
    )

    # Đóng gói các tham số cố định cho hàm objective bằng lambda
    # Pass necessary variables from run_optimization's scope into the objective function
    objective_func = lambda trial: objective(trial, args, node_attr, edge_attr, out_dim, train_loader, val_loader, device)

    print(f"Running Optuna optimization for {args.n_trials} trials...")
    try:
        study.optimize(objective_func, n_trials=args.n_trials, timeout=args.timeout) # Thêm timeout nếu muốn giới hạn thời gian tổng
    except KeyboardInterrupt:
         print("Optimization stopped by user.")
    # Catch potential issues during optimization itself (less common than trial errors)
    except Exception as opt_e:
         print(f"An error occurred during the optimization process: {opt_e}")


    # --- In và Lưu kết quả tốt nhất ---
    print("\n" + "="*30 + " Optuna Optimization Finished " + "="*30)
    try:
        print(f"Study '{study.study_name}' finished with {len(study.trials)} trials.")

        # Check if any trials completed successfully before accessing best_trial
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed_trials:
             print("No trials completed successfully. Cannot determine best parameters.")
             # Optionally, print info about failed trials
             failed_trials = [t for t in study.trials if t.state != optuna.trial.TrialState.COMPLETE]
             print(f"Number of failed/pruned trials: {len(failed_trials)}")
             # You could add more detailed logging about why trials failed here if needed
        else:
             best_trial = study.best_trial
             best_params = best_trial.params
             best_value = best_trial.value
             print(f"Best validation loss found: {best_value:.4f} (Trial {best_trial.number})")
             print("Best hyperparameters found:")
             print(json.dumps(best_params, indent=4))

             # Lưu best_params vào file json
             best_params_path = os.path.join(args.output_dir, f"{study_name}_best_params.json")
             Path(args.output_dir).mkdir(parents=True, exist_ok=True)
             with open(best_params_path, "w") as f:
                 json.dump(best_params, f, indent=4)
             print(f"Best parameters saved to {best_params_path}")

    # Catch the specific error when no successful trials are found
    except ValueError as e:
        if "contains no completed trials" in str(e):
            print(f"Optuna study '{study.study_name}' finished, but no trials completed successfully.")
        else:
            print(f"An unexpected ValueError occurred retrieving results: {e}")
    except Exception as e:
         print(f"An unexpected error occurred retrieving results: {e}")


# --- Parser cho Command Line Arguments ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Optuna Hyperparameter Optimization for GNN")

    # Basic data & model parameters
    parser.add_argument("--data_path", type=str, default='./Data/regression/lograte/lograte.csv')
    parser.add_argument("--graph_path", type=str, default='./Data/regression/lograte/its_new/lograte.pkl.gz')
    parser.add_argument("--y_column", type=str, default="lograte")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=27407)

    # Optuna parameters
    parser.add_argument("--n_trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--n_epochs_per_trial", type=int, default=15, help="Number of epochs per Optuna trial")
    parser.add_argument("--study_name", type=str, default="gnn_optimization", help="Name for the Optuna study")
    parser.add_argument("--db_path", type=str, default="./optuna_db/", help="Directory to save Optuna study database")
    parser.add_argument("--output_dir", type=str, default="./optuna_results/", help="Directory to save best parameters JSON")
    parser.add_argument("--timeout", type=int, default=None, help="Optional timeout for Optuna study in seconds")

    # Optuna search space parameters (passed via args)
    parser.add_argument('--min_depth', type=int, default=2)
    parser.add_argument('--max_depth', type=int, default=6)
    parser.add_argument('--node_hid_choices', type=int, nargs='+', default=[128, 256, 300, 512])
    # parser.add_argument('--edge_hid_choices', type=int, nargs='+', default=[128, 256, 300, 512]) # Uncomment if edge_hid is used
    parser.add_argument('--min_dr', type=float, default=0.0)
    parser.add_argument('--max_dr', type=float, default=0.5)

    # Add any args needed by the 'inference' function if they are not already present
    # Example: If inference needs a specific metric name
    # parser.add_argument("--metric_name", type=str, default="RMSE", help="Metric for inference reporting")

    args = parser.parse_args()

    # Ensure paths exist or handle errors
    if not Path(args.data_path).is_file():
        print(f"Error: Data file not found at {args.data_path}")
        sys.exit(1)
    if not Path(args.graph_path).is_file():
        print(f"Error: Graph file not found at {args.graph_path}")
        sys.exit(1)

    # Run the optimization
    run_optimization(args)
