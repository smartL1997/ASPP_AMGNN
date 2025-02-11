import argparse
import glob
import time
import random
import json
import os

os.environ["WANDB_API_KEY"] = ""
os.environ["WANDB_MODE"] = "offline"

import wandb
wandb.login()
import numpy as np
from tqdm import tqdm
from typing import Tuple
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from  models.ASPP_AMGCN import Model
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score
from utils.metrics import Metrics, AUC, metric_summary
from utils.loss import common_loss, loss_dependence
from utils import ecg_data

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--train', type=bool, default=True, help='train and valid')
parser.add_argument('--test', type=bool, default=True, help='test')
parser.add_argument('--epochs', type=int, default=60, help='maximum number of epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lrscheduler', type=list, default=[5, 10, 15, 20, 25], help='learning rate')

parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')

parser.add_argument('--sample_neighbor', type=bool, default=True, help='whether sample neighbors')
parser.add_argument('--sparse_attention', type=bool, default=True, help='whether use sparse attention')
parser.add_argument('--structure_learning', type=bool, default=True, help='whether perform structure learning')
parser.add_argument('--pooling_ratio', type=float, default=0.5, help='pooling ratio')
parser.add_argument('--lamb', type=float, default=1.0, help='trade-off parameter')

parser.add_argument('--patience', type=int, default=5, help='patience for early stopping')
parser.add_argument('--nhid', type=int, default=64, help='hidden size')

parser.add_argument("--model", type=str, default="ASPP_AMGCN", help="Select the model to train")
parser.add_argument('--dataset', type=str, default='ptb', help='ptb/ICBEB')
parser.add_argument("--loggr", type=bool, default=True, help="Enable wandb logging")
# dep 5e-11
parser.add_argument("--beta", type=float, default=5e-11, help="beta of loss dependence")
# com 0.01
parser.add_argument("--theta", type=float, default=0.01, help="theta of loss common loss")

args = parser.parse_args()

# Random seed
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.enable = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


beta = args.beta
theta = args.theta

def print_datainfo(dataset):
    print(f'Dataset: {dataset}:')
    print('====================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    data = dataset[0]  # Get the first graph object.
    print()
    print(data)
    print('=============================================================')
    # Gather some statistics about the first graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train_epoch(
        model: nn.Module,
        optimizer: torch.optim,
        loss_func,
        train_loader,
        epoch: int,
        loggr: None,
) -> Tuple[float, float, float]:

    model.train()
    pred_all = []
    loss_all = []
    gt_all = []

    for _, data in tqdm(enumerate(train_loader), total=len(train_loader), desc="train"):
        data = data.to(device)
        pred, att, emb1, com1, com2, emb2, emb = model(data)

        pred_all.append(pred.cpu().detach().numpy())
        y_true = torch.tensor(np.array(data.y), dtype=torch.double)

        # 构造损失函数
        loss_dep = (loss_dependence(emb1, com1) + loss_dependence(emb2, com2)) / 2
        loss_com = common_loss(com1, com2)
        loss_dep_beta = beta * loss_dep
        loss_com_theta = theta * loss_com
        loss_class = loss_func(pred, y_true.to(device))
        loss = loss_class + loss_dep_beta + loss_com_theta
        loss_all.append(loss.cpu().detach().item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        gt_all.append(y_true.cpu().detach().numpy())

    print("Epoch: {0}".format(epoch))
    pred_all = np.concatenate(pred_all, axis=0)
    gt_all = np.concatenate(gt_all, axis=0)
    _, mean_acc = Metrics(np.array(gt_all), pred_all)
    roc_score = roc_auc_score(np.array(gt_all), pred_all, average="macro")


    if loggr is not None:
        loggr.log({"train_mean_accuracy": mean_acc, "epoch": epoch})
        loggr.log({"train_roc_score": roc_score, "epoch": epoch})
        loggr.log({"train_loss": np.mean(loss_all), "epoch": epoch})

    print(f'train_loss: {np.mean(loss_all)}, train_mean_accuracy: {mean_acc},train_roc_score: {roc_score}')

    return np.mean(loss_all), mean_acc, roc_score


def test_epoch(
        model: nn.Module,
        loss_func: torch.optim,
        loader,
        epoch: int,
        loggr: None,
) -> Tuple[float, float, float]:
    model.eval()

    pred_all = []
    loss_all = []
    gt_all = []

    for _, data in tqdm(enumerate(loader), total=len(loader), desc="valid"):
        data = data.to(device)
        pred, att, emb1, com1, com2, emb2, emb = model(data)
        y_true = torch.tensor(np.array(data.y), dtype=torch.double)

        # 构造损失函数
        loss_dep = (loss_dependence(emb1, com1) + loss_dependence(emb2, com2)) / 2
        loss_com = common_loss(com1, com2)
        loss_dep_beta = beta * loss_dep
        loss_com_theta = theta * loss_com
        loss_class = loss_func(pred, y_true.to(device))
        loss = loss_class + loss_dep_beta + loss_com_theta

        pred_all.append(pred.cpu().detach().numpy())
        gt_all.append(y_true.cpu().detach().numpy())
        loss_all.append(loss.cpu().detach().numpy())

    pred_all = np.concatenate(pred_all, axis=0)
    gt_all = np.concatenate(gt_all, axis=0)
    _, mean_acc = Metrics(np.array(gt_all), pred_all)
    roc_score = roc_auc_score(np.array(gt_all), pred_all, average="macro")

    if loggr is not None:
        loggr.log({"test_mean_accuracy": mean_acc, "epoch": epoch})
        loggr.log({"test_roc_score": roc_score, "epoch": epoch})
        loggr.log({"test_loss": np.mean(loss_all), "epoch": epoch})

    print(f'test_loss: {np.mean(loss_all)}, test_mean_accuracy: {mean_acc},test_roc_score: {roc_score}')

    return np.mean(loss_all), mean_acc, roc_score


def train(
        loggr,
        train_loader,
        test_loader,
        model: nn.Module,
        epochs: int = 60,
        name: str = "MSAGFN",
) -> None:

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.lrscheduler, gamma=0.5)
    loss_func = torch.nn.BCEWithLogitsLoss()
    best_score = 0.0

    patience = args.patience
    epochs_without_improvement = 0
    for epoch in range(epochs):

        # 训练 train_loader
        train_results = train_epoch(model, optimizer, loss_func, train_loader, epoch, loggr=loggr,)
        test_results = test_epoch(model, loss_func, test_loader, epoch, loggr=loggr)
        scheduler.step()
        print("<==================================================>")

        score = test_results[2]
        if epoch >= 3 and best_score <= score:
            best_score = score
            save_path = os.path.join(os.getcwd(), "checkpoints/", f"{name}_weights.pt")
            torch.save(model.state_dict(), save_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping. No improvement for {epoch} epochs")
            # break



def test(
        model: nn.Module,
        test_loader,
        num_classes
) -> None:

    pred_all = []
    loss_all = []
    gt_all = []

    model.eval()

    for i, data in tqdm(enumerate(test_loader), total=len(test_loader), desc="test"):
        data = data.to(device)
        pred, _, _, _, _, _, _ = model(data)
        y_true = torch.tensor(np.array(data.y), dtype=torch.double)
        pred_all.append(pred.cpu().detach().numpy())
        gt_all.append(y_true.cpu().detach().numpy())

    pred_all = np.concatenate(pred_all, axis=0)
    y_test = np.array(np.concatenate(gt_all, axis=0))
    roc_score = roc_auc_score(y_test, pred_all, average="macro")
    acc, mean_acc = Metrics(y_test, pred_all)
    class_auc = AUC(y_test, pred_all)
    summary = metric_summary(y_test, pred_all)

    print(f"class wise accuracy: {acc}")
    print(f"accuracy: {mean_acc}")
    print(f"roc_score : {roc_score}")
    print(f"class wise AUC : {class_auc}")
    print(f"class wise precision, recall, f1 score : {summary}")

    logs = dict()
    # overall
    logs["roc_score"] = roc_score.tolist()
    logs["mean_acc"] = mean_acc
    # every class
    logs["accuracy"] = acc
    logs["class_auc"] = class_auc
    logs["class_precision_recall_f1"] = summary
    name = "output"
    logs_path = os.path.join(os.getcwd(), "logs", f"{name}_logs.json")
    jsObj = json.dumps(logs)
    fileObject = open(logs_path, 'w')
    fileObject.write(jsObj)
    fileObject.close()

    # roc_score, mean_acc 和 F1
    return roc_score, mean_acc, summary[0]


if __name__ == "__main__":

    # data_root = "./data/"
    data_root = "F:/chenteng/ECGNN-main/data/"

    sampling_rate = 100
    # Load raw ecg data
    data_path = os.path.join(data_root, args.dataset, "raw/")
    _, _, Y = ecg_data.load_dataset(data_path, sampling_rate)

    # Build graph datasets
    ecg_dataset = ecg_data.ECGDataset(os.path.join(data_root, args.dataset))
    print_datainfo(ecg_dataset)
    args.num_classes = 5
    args.num_features = ecg_dataset.num_features

    # Split dataset
    train_dataset, val_dataset, test_dataset = ecg_data.select_dataset(ecg_dataset, Y)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Build Model
    model = Model(args).double().to(device)

    # Logging
    if args.loggr:
        wandb = wandb.init(
            project="final_paper",
            name=args.model,
            notes=f"k=3 and beta: {args.beta} and theta: {args.theta} and batch size: {args.batch_size} and epochs: {args.epochs}and lr:{args.lr} and lr_schedule:{args.lrscheduler} ",
            save_code=True,
        )
        args.logger = wandb

    # Model training
    if args.train:
        print("\n<=============== Start Training ===============>\n")
        train(
            loggr=args.logger,
            model=model,
            epochs=args.epochs,
            name=args.model,
            train_loader=train_loader,
            test_loader=val_loader,
        )

    # Model testing
    if args.test:
        print("<=============== Start Testing ===============>")
        path_weights = os.path.join(os.getcwd(), "checkpoints", f"{args.model}_weights.pt")
        model.load_state_dict(torch.load(path_weights))

        best_test_roc_score, test_mearn_acc, F1 = test(model, test_loader, num_classes=5)

        args.logger.summary["best_test_roc_score"] = best_test_roc_score
        args.logger.summary["test_mearn_acc"] = test_mearn_acc
        args.logger.summary["F1"] = F1

    args.logger.finish()












