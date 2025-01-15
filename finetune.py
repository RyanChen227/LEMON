import os
import torch
import random
import argparse
import numpy as np
import pandas as pd
import os.path as osp
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import GNN_graphpred
import torch.nn.functional as F
from loader import MoleculeDataset
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader
from splitters import random_split, scaffold_split, random_scaffold_split


criterion = nn.BCEWithLogitsLoss(reduction = "none")


def train(args, model, device, loader, optimizer):
    model.train()
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch)
        y = batch.y.view(pred.shape).to(torch.float64)
        #Whether y is non-null or not.
        is_valid = y**2 > 0
        #Loss matrix
        loss_mat = criterion(pred.double(), (y+1)/2)
        #loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
        optimizer.zero_grad()
        loss = torch.sum(loss_mat)/torch.sum(is_valid)
        loss.backward()
        optimizer.step()


def eval(args, model, device, loader):
    model.eval()
    y_true = []
    y_scores = []
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch)
        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)
    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()
    roc_list = []
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
            is_valid = y_true[:,i]**2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))
    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))
    return np.array(roc_list).mean()


def main(args):
    random.seed(args.runseed)
    np.random.seed(args.runseed)
    torch.manual_seed(args.runseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    #Bunch of classification tasks
    if args.dataset == "tox21":
        num_tasks = 12
    elif args.dataset == "hiv":
        num_tasks = 1
    elif args.dataset == "pcba":
        num_tasks = 128
    elif args.dataset == "muv":
        num_tasks = 17
    elif args.dataset == "bace":
        num_tasks = 1
    elif args.dataset == "bbbp":
        num_tasks = 1
    elif args.dataset == "toxcast":
        num_tasks = 617
    elif args.dataset == "sider":
        num_tasks = 27
    elif args.dataset == "clintox":
        num_tasks = 2
    else:
        raise ValueError("Invalid dataset name.")

    #set up dataset
    dpath = 'chem/'
    dataset = MoleculeDataset(osp.join(dpath, args.dataset), dataset=args.dataset)
    if args.split == "scaffold":
        smiles_list = pd.read_csv(osp.join(dpath, args.dataset, 'processed/smiles.csv'), header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)
        print("scaffold")
    elif args.split == "random":
        train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
        print("random")
    elif args.split == "random_scaffold":
        smiles_list = pd.read_csv(osp.join(dpath, args.dataset, 'processed/smiles.csv'), header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
        print("random scaffold")
    else:
        raise ValueError("Invalid split option.")

    fb_keys = ['lg_x']
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, follow_batch=fb_keys)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, follow_batch=fb_keys)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, follow_batch=fb_keys)

    #set up model
    model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK=args.JK,
                          drop_ratio=args.dropout_ratio, graph_pooling=args.graph_pooling,
                          gnn_type=args.gnn_type, eval_type=args.eval_type)
    model_path =  'models/%s_a%s_b%s.pt' % (args.model_epoch, args.alpha, args.beta)
    if not os.path.exists(model_path):
        print('Not exist:\t', model_path)
        return
    model.from_pretrained(model_path)
    model.to(device)

    #set up optimizer
    model_param_group = []
    model_param_group.append({"params": model.gnn.parameters()})
    if args.graph_pooling == "attention":
        model_param_group.append({"params": model.pool.parameters(), "lr":args.lr*args.lr_scale})
    model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr":args.lr*args.lr_scale})
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)

    val_acc_list = []
    test_acc_list = []

    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
        train(args, model, device, train_loader, optimizer)
        val_acc = eval(args, model, device, val_loader)
        test_acc = eval(args, model, device, test_loader)
        print("%s %s val: %.4f test: %.4f" % (args.dataset, args.runseed,
                                              val_acc*100, test_acc*100), flush=True)
        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)
    val_acc_list = np.array(val_acc_list)
    test_acc_list = np.array(test_acc_list)
    val_max_idx = val_acc_list.argmax()
    with open(log_path, 'a+') as fp:
        fp.write('%s %s %s %s\t%.6f %.6f %.6f %.6f %.6f\n' % (
                 args.model_epoch, args.eval_type, args.alpha, args.beta,
                 val_acc_list.max(), test_acc_list.max(), test_acc_list[val_max_idx],
                 val_acc, test_acc))


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--dataset', type=str, default = 'tox21',
                        help='root directory of dataset. For now, only classification.')
    parser.add_argument('--model_epoch', type=int, default=100,
                        help='filename to read the model (if there is any)')
    parser.add_argument('--seed', type=int, default=42,
                        help="Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=0,
                        help="Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type = str, default="scaffold",
                        help="random or scaffold or random_scaffold")
    parser.add_argument('-et', '--eval_type', type=str, default='hybird', choices=['graph', 'line_graph', 'hybird'],
                        help='evaluating with which embedding or both')
    args = parser.parse_args()

    main(args)
