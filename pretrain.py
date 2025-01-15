import sys
import torch
import random
import argparse
import numpy as np
import os.path as osp
from model import GNN
import torch.nn as nn
from tqdm import tqdm
from copy import deepcopy
import torch.optim as optim
import torch.nn.functional as F
from loader import MoleculeDataset
from torch_scatter import scatter_add
from loader import MoleculeDataset_aug
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool


class graphcl(nn.Module):
    def __init__(self, gnn):
        super(graphcl, self).__init__()
        self.gnn = gnn
        self.pool = global_mean_pool
        self.node_phead = nn.Sequential(nn.Linear(300, 300), nn.ReLU(inplace=True), nn.Linear(300, 300))
        self.edge_phead = nn.Sequential(nn.Linear(300, 300), nn.ReLU(inplace=True), nn.Linear(300, 300))
        self.node_transform = nn.Linear(300*2, 300)

    def forward_cl(self, batch):
        h_list, e_list = self.gnn(batch)
        node_rep, edge_rep = h_list[-1], e_list[-1]
        x1 = self.node_phead(self.pool(node_rep, batch.batch))
        x2 = self.edge_phead(self.pool(edge_rep, batch.lg_x_batch))
        return x1, x2, node_rep, edge_rep

    def local_cl(self, edge_rep, node_rep, lg_edge_index_map, batch):
        # concat node to form edge representation
        edge_n1 = node_rep[lg_edge_index_map[0]]
        edge_n2 = node_rep[lg_edge_index_map[1]]
        edge_rep2 = torch.cat([edge_n1, edge_n2], dim=1)
        edge_rep2 = self.node_transform(edge_rep2)
        x1, x2 = edge_rep, edge_rep2

        T = 0.1
        N = x1.size(0)
        G = batch.max() + 1
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(N), range(N)]
        pos_sim = scatter_add(pos_sim, batch)

        batch_sim = scatter_add(scatter_add(sim_matrix, batch, dim=0), batch, dim=1) 
        graph_sim = batch_sim[range(G), range(G)]

        # inter-local contrastive loss
        neg_sim0 = batch_sim.sum(dim=0) - graph_sim
        neg_sim1 = batch_sim.sum(dim=1) - graph_sim
        loss0 = pos_sim / neg_sim0
        loss1 = pos_sim / neg_sim1
        loss0 = -torch.log(loss0).mean()
        loss1 = -torch.log(loss1).mean()
        inter_loss = (loss0 + loss1) / 2.0

        # intra-local contrastive loss
        neg_sim = graph_sim - pos_sim
        loss = pos_sim / neg_sim
        inner_loss = -torch.log(loss).mean()

        return inter_loss, inner_loss

    def global_cl(self, x1, x2):
        T = 0.1
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim)
        loss1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss0 = -torch.log(loss0).mean()
        loss1 = -torch.log(loss1).mean()
        loss = (loss0 + loss1) / 2.0
        return loss

    def loss_cl(self, x1, x2, args, edge_rep, node_rep, lg_edge_index_map, batch):
        gloss = self.global_cl(x1, x2)
        inter_loss, inner_loss = self.local_cl(edge_rep, node_rep, lg_edge_index_map, batch)
        loss = gloss + args.alpha*inter_loss + args.beta*inner_loss
        return loss


def train(args, model, device, loader, optimizer, loss_sym=True):
    model.train()
    train_loss_accum = 0
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        optimizer.zero_grad()
        x1, x2, node_rep, edge_rep = model.forward_cl(batch)
        loss = model.loss_cl(x1, x2, args, edge_rep, node_rep, batch.lg_edge_index_map, batch.lg_x_batch)
        loss.backward()
        optimizer.step()
        train_loss_accum += float(loss.detach().cpu().item())
    return train_loss_accum/(step+1)


def main(args):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    #set up dataset
    dpath = 'chem/'
    dataset = MoleculeDataset(osp.join(dpath, args.dataset), dataset=args.dataset)
    fb_keys = ['lg_x']
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, follow_batch=fb_keys)
    #set up model
    gnn = GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type)
    model = graphcl(gnn).to(device)
    #set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

    for epoch in range(1, args.epochs+1):
        print(epoch, flush=True)
        train_loss = train(args, model, device, loader, optimizer, args.loss_sym)
        print(train_loss, flush=True)
        if epoch % 20 == 0:
            model_path =  'models/%s_a%s_b%s' % (epoch, args.alpha, args.beta)
            torch.save(model.gnn.state_dict(), model_path + '.pt')


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--alpha', type=float, default=1,
                        help='a tuning parameter for inter local loss')
    parser.add_argument('--beta', type=float, default=1,
                        help='a tuning parameter for inner local loss')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('-d', '--dataset', type=str, default='zinc_standard_agent',
                        help='root directory of dataset. For now, only classification.')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--seed', type=int, default=0,
                        help = "Seed for splitting dataset.")
    parser.add_argument('--loss_sym', action='store_true', default=True)
    args = parser.parse_args()

    main(args)
