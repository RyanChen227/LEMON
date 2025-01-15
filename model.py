import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set

num_atom_type = 120  # including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6  # including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3


class GINConv(MessagePassing):
    '''
    Extension of GIN aggregation to incorporate edge information by concatenation.
    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not.
    See https://arxiv.org/abs/1810.00826
    '''
    def __init__(self, emb_dim, aggr='add', is_lg=False):
        super(GINConv, self).__init__()
        self.aggr = aggr
        self.is_lg = is_lg
        self.emb_dim = emb_dim
        #multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        # edge embedding
        self.edge_embedding1 = torch.nn.Embedding(num_atom_type if is_lg else num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_chirality_tag if is_lg else num_bond_direction, emb_dim)
        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index, edge_weight = add_self_loops(edge_index, num_nodes=x.size(0))
        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        if self.is_lg:
            self_loop_attr[:,0] = 119  # atom type for self-loop edge
        else:
            self_loop_attr[:,0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(torch.int64)
        self_loop_attr = self.edge_embedding1(self_loop_attr[:,0]) + self.edge_embedding2(self_loop_attr[:,1])
        if edge_attr.size(1) != self.emb_dim:
            edge_attr = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GNN(torch.nn.Module):
    '''
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat
    Output:
        node representations
    '''
    def __init__(self, num_layer, emb_dim, JK='last', drop_ratio=0, gnn_type='gin'):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError('Number of GNN layers must be greater than 1.')

        # node embedding
        self.node_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.node_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)
        torch.nn.init.xavier_uniform_(self.node_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.node_embedding2.weight.data)
        # edge embedding
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)
        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        ### graph
        self.gnns = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GINConv(emb_dim, aggr='add'))
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))
        ### line graph
        self.lg_gnns = torch.nn.ModuleList()
        self.lg_batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.lg_gnns.append(GINConv(emb_dim, aggr='add', is_lg=True))
            self.lg_batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        lg_x, lg_edge_index = data.lg_x, data.lg_edge_index
        lg_edge_attr = data.x[data.lg_edge_index_map2]

        x = self.node_embedding1(x[:,0]) + self.node_embedding2(x[:,1])
        lg_x = self.edge_embedding1(lg_x[:,0]) + self.edge_embedding2(lg_x[:,1])

        h_list, e_list = [x], [lg_x]
        for layer in range(self.num_layer):
            # graph
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer < self.num_layer-1:
                h = F.relu(h)
            h = F.dropout(h, self.drop_ratio, training=self.training)
            h_list.append(h)
            # line graph
            e = self.lg_gnns[layer](e_list[layer], lg_edge_index, lg_edge_attr)
            e = self.lg_batch_norms[layer](e)
            if layer < self.num_layer-1:
                e = F.relu(e)
            e = F.dropout(e, self.drop_ratio, training=self.training)
            e_list.append(e)
            # update edge attr
            edge_index = torch.cat([data.lg_edge_index_map, data.lg_edge_index_map[[1, 0]]], dim=1)
            edge_attr = torch.cat([e, e], dim=0)
            lg_edge_attr = h[data.lg_edge_index_map2]

        return h_list, e_list


class GNN_graphpred(torch.nn.Module):
    '''
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set
        gnn_type: gin, gcn, graphsage, gat

    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    '''
    def __init__(self, num_layer, emb_dim, num_tasks, JK='last', drop_ratio=0, graph_pooling='mean', gnn_type='gin', eval_type='hybird'):
        super(GNN_graphpred, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.eval_type = eval_type

        if self.num_layer < 2:
            raise ValueError('Number of GNN layers must be greater than 1.')

        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type=gnn_type)

        #Different kind of graph pooling
        if graph_pooling == 'sum':
            self.pool = global_add_pool
        elif graph_pooling == 'mean':
            self.pool = global_mean_pool
        elif graph_pooling == 'max':
            self.pool = global_max_pool
        elif graph_pooling == 'attention':
            if self.JK == 'concat':
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear((self.num_layer+1)*emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == 'set2set':
            set2set_iter = int(graph_pooling[-1])
            if self.JK == 'concat':
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError('Invalid graph pooling type.')

        #For graph-level binary classification
        mult = 2 if graph_pooling[:-1] == 'set2set' else 1
        if eval_type == 'hybird':
            mult *= 2

        if self.JK == 'concat':
            if eval_type == 'hybird':
                self.graph_pred_linear = torch.nn.Sequential(torch.nn.Linear(mult*(self.num_layer+1)*emb_dim, emb_dim), torch.nn.ReLU(), torch.nn.Linear(emb_dim, num_tasks))
            else:
                self.graph_pred_linear = torch.nn.Linear(mult*(self.num_layer+1)*emb_dim, num_tasks)
        else:
            if eval_type == 'hybird':
                self.graph_pred_linear = torch.nn.Sequential(torch.nn.Linear(mult*emb_dim, emb_dim), torch.nn.ReLU(), torch.nn.Linear(emb_dim, num_tasks))
            else:
                self.graph_pred_linear = torch.nn.Linear(mult*emb_dim, num_tasks)

    def from_pretrained(self, model_file):
        self.gnn.load_state_dict(torch.load(model_file))

    def forward(self, data):
        batch, e_batch = data.batch, data.lg_x_batch
        h_list, e_list = self.gnn(data)
        if self.JK == 'concat':
            node_rep = torch.cat(h_list, dim=1)
            edge_rep = torch.cat(e_list, dim=1)
        else:
            node_rep, edge_rep = h_list[-1], e_list[-1]
        node_pool = self.pool(node_rep, batch)
        edge_pool = self.pool(edge_rep, e_batch)
        if self.eval_type == 'hybird':
            out = self.graph_pred_linear(torch.cat([node_pool, edge_pool], dim=1))
        elif self.eval_type == 'line_graph':
            out = self.graph_pred_linear(edge_pool)
        else:
            out = self.graph_pred_linear(node_pool)
        return out


if __name__ == '__main__':
    pass

