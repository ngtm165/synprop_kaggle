import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import os
import sys
root_dir=Path(__file__).resolve().parents[1]
sys.path.append(str(root_dir))
os.chdir(str(root_dir))
from synprop.model_GIN import model, train, inference
from synprop.data_wrapper_7 import data_wrapper_7

def finetune(args):
    batch_size = args.batch_size
    model_path = args.model_path + args.model_name
    monitor_path = args.monitor_folder + args.monitor_name
    epochs = args.epochs
    data_path =args.data_path
    graph_path=args.graph_path
    target=args.y_column

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

    data_loader = data_wrapper_7(data_path, graph_path, target,  batch_size,4, 0.1, 0.1) #ver 6,7
    train_loader, val_loader, test_loader = data_loader.get_data_loaders()
    node_attr=train_loader.dataset[0].x.shape[1]
    # edge_attr=train_loader.dataset[0].edge_attr.shape[1]

    print("--- model_path:", model_path)

    # training
    if not os.path.exists(model_path):
        net = model(node_attr).to(device) #thử bỏ edge_attr
        print("-- TRAINING")
        net = train(
            args, net, train_loader, val_loader, model_path, device, epochs=epochs
        )
    else:
        net = model(node_attr).to(device) #thử bỏ edge_attr
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint["model_state_dict"])
        current_epoch = checkpoint["epoch"]
        epochs = epochs - current_epoch
        net = train(
            args,
            net,
            train_loader,
            val_loader,
            model_path,
            device,
            epochs=epochs,
            current_epoch=current_epoch,
            best_val_loss=checkpoint["val_loss"],
        )

    # test
    net = model(node_attr).to(device) #thử bỏ edge_attr
    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint["model_state_dict"])
    rmse, mae = inference(args, net, test_loader, device)
    print("-- RESULT")
    print("--- rmse: %.3f, MAE: %.3f," % (rmse, mae))
    dict = {
        "Name": "Test",
        "test_rmse": rmse,
        "test_mae": mae,
    }
    with open(monitor_path, "a") as f:
        f.write(json.dumps(dict) + "\n")




