import argparse
import os
import random
import numpy as np
import torch
from pathlib import Path
import os
import sys

root_dir=Path(__file__).resolve().parents[1]
sys.path.append(str(root_dir))
os.chdir(str(root_dir))
from synprop.finetune_graph import finetune


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--batch_size", type=int, default=16)
    arg_parser.add_argument("--epochs", type=int, default=100)
    arg_parser.add_argument("--device", type=int, default=0)
    arg_parser.add_argument("--monitor_folder", type=str, default="./Data/monitor/")
    arg_parser.add_argument("--monitor_name", type=str, default="monitor.txt")
    arg_parser.add_argument("--Data_folder", type=str, default="./Data/")
    arg_parser.add_argument("--model_path", type=str, default="./Data/model/")
    arg_parser.add_argument("--data_path", type=str, default='./Data/regression/lograte/lograte.csv')
    # arg_parser.add_argument("--column_rxn", type=str, default="AAM") #ver3,4,5
    # arg_parser.add_argument("--reaction_mode_str", type=str, default="reac_diff") #mới cho ver4   
    arg_parser.add_argument("--graph_path", type=str, default='./Data/regression/lograte/its_origin/lograte.pkl.gz') #binh thuong bo cho nay
    arg_parser.add_argument("--model_name", type=str, default="model.pt")
    arg_parser.add_argument("--y_column", type=str, default="lograte")
    arg_parser.add_argument("--seed", type=int, default=27407)
    args = arg_parser.parse_args()

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False

    finetune(args)
