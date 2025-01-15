#!/usr/bin/env python
# encoding: utf-8
import torch
from torch_sparse import coalesce
from torch_scatter import scatter_add
from torch_geometric.data import Data
from torch_geometric.typing import OptTensor
from torch_geometric.utils import remove_self_loops
from torch_geometric.transforms import BaseTransform


class LGData(Data):
    # for line graph
    def __init__(self, x: OptTensor=None, edge_index: OptTensor=None,
                 edge_attr: OptTensor=None, y: OptTensor=None,
                 pos: OptTensor=None, **kwargs):
        super().__init__(x, edge_index, edge_attr, y, pos, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'lg_edge_index':
            return self.lg_num_nodes
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        return super().__cat_dim__(key, value, *args, **kwargs)


class LineGraph(BaseTransform):
    r"""Converts a graph to its corresponding line-graph:

    .. math::
        L(\mathcal{G}) &= (\mathcal{V}^{\prime}, \mathcal{E}^{\prime})

        \mathcal{V}^{\prime} &= \mathcal{E}

        \mathcal{E}^{\prime} &= \{ (e_1, e_2) : e_1 \cap e_2 \neq \emptyset \}

    Line-graph node indices are equal to indices in the original graph's
    coalesced :obj:`edge_index`.
    For undirected graphs, the maximum line-graph node index is
    :obj:`(data.edge_index.size(1) // 2) - 1`.

    New node features are given by old edge attributes.
    For undirected graphs, edge attributes for reciprocal edges
    :obj:`(row, col)` and :obj:`(col, row)` get summed together.

    Args:
        force_directed (bool, optional): If set to :obj:`True`, the graph will
            be always treated as a directed graph. (default: :obj:`False`)
    """
    def __init__(self, force_directed=False):
        self.force_directed = force_directed

    def __call__(self, data):
        lgd = LGData()
        # copy graph data
        for key in data.keys:
            lgd[key] = data[key]
        # no edge
        if lgd.edge_index.shape[1] == 0:
            lgd.lg_edge_index_map = torch.tensor([range(lgd.num_nodes), range(lgd.num_nodes)],
                                                 dtype=data.edge_index.dtype)  # lg_node = g_edge
            lgd.lg_edge_index_map2 = torch.tensor([], dtype=data.edge_index.dtype) # lg_edge == g_node
            lgd.lg_x = torch.zeros(lgd.num_nodes, 2, dtype=lgd.x.dtype)
            lgd.lg_edge_index = lgd.edge_index
            lgd.lg_num_nodes = lgd.num_nodes
            return lgd

        N = data.num_nodes
        edge_index, edge_attr = data.edge_index, data.edge_attr
        (row, col), edge_attr = coalesce(edge_index, edge_attr, N, N)

        # Compute node indices.
        mask = row < col
        row, col = row[mask], col[mask]
        i = torch.arange(row.size(0), dtype=torch.long, device=row.device)
        # store the map of edge_node in line graph, node of line graph == edge of ori graph
        lgd.lg_edge_index_map = torch.stack([row, col], dim=0)

        (row, col), i = coalesce(
            torch.stack([
                torch.cat([row, col], dim=0),
                torch.cat([col, row], dim=0)
            ], dim=0), torch.cat([i, i], dim=0), N, N)

        # Compute new edge indices according to `i`.
        count = scatter_add(torch.ones_like(row), row, dim=0,
                            dim_size=data.num_nodes)
        joints = torch.split(i, count.tolist())

        def generate_grid(x):
            row = x.view(-1, 1).repeat(1, x.numel()).view(-1)
            col = x.repeat(x.numel())
            return torch.stack([row, col], dim=0)

        joints = [generate_grid(joint) for joint in joints]
        joints = torch.cat(joints, dim=1)
        joints, _ = remove_self_loops(joints)
        N = row.size(0) // 2
        joints, _ = coalesce(joints, None, N, N)

        if edge_attr is not None:
            lgd.lg_x = scatter_add(edge_attr, i, dim=0, dim_size=N) / 2
            lgd.lg_x = lgd.lg_x.to(dtype=lgd.x.dtype)
        lgd.lg_edge_index = joints
        lgd.lg_num_nodes = edge_index.size(1) // 2
        # edge of line graph == node of ori graph
        edge1 = lgd.lg_edge_index_map[:,joints[0]]
        edge2 = lgd.lg_edge_index_map[:,joints[1]]
        map2 = torch.zeros(lgd.lg_edge_index.size(1), dtype=lgd.edge_index.dtype)
        for i in range(2):
            for j in range(2):
                map2 += edge1[i] * (edge1[i] == edge2[j])
        lgd.lg_edge_index_map2 = map2
        return lgd

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
