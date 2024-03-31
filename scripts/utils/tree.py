import torch 
import dgl
import numpy as np

import warnings
warnings.filterwarnings("ignore", message="Recommend creating")

"""
Reference: https://github.com/microsoft/icecaps/blob/master/icecaps/util/trees.py
           https://github.com/inyukwo1/tree-lstm/blob/master/tree_lstm/tree_lstm.py (2023/07/07)
"""

class FragmentTree:
    def __init__(self, dgl_graph= None):
        self.dgl_graph = dgl_graph if dgl_graph else dgl.DGLGraph()
        self.max_depth = 0
        self.max_degree = 0

    def add_node(self, parent_id=None, feature: torch.Tensor = torch.Tensor(), fid: int= -1, bondtype: int= 1, data: dict= None):
        if data is None:
            data = {'x': feature.unsqueeze(0),
                    'fid': torch.tensor([fid]).unsqueeze(0)}
        self.dgl_graph.add_nodes(1, data= data)
        added_node_id = self.dgl_graph.number_of_nodes() - 1

        if parent_id is not None:
            self.dgl_graph.ndata['level'][added_node_id] = self.dgl_graph.ndata['level'][parent_id] + 1
            self.dgl_graph.add_edges(added_node_id, parent_id, data= {'w': torch.tensor([bondtype]).unsqueeze(0)})
            self.max_degree = max(self.max_degree, len(self.dgl_graph.predecessors(parent_id)))
        elif added_node_id > 0:
            self.dgl_graph.ndata['level'][added_node_id] = torch.tensor([0]).int()
        else:
            self.dgl_graph.ndata['level'] = torch.tensor([0]).int()

        self.max_depth = self.dgl_graph.ndata['level'].max().item()
        # self.max_width = max([self.width(level) for level in range(self.max_depth+1)]).item()

        return added_node_id

    def add_link(self, child_id, parent_id, bondtype: int= 1):
        self.dgl_graph.add_edges(child_id, parent_id, data= {'w': torch.tensor([bondtype]).unsqueeze(0)})

    def remove_node(self, node_id: int):
        self.dgl_graph.remove_nodes(node_id)

    def remove_edge(self, edge_id: int):
        self.dgl_graph.remove_edges(edge_id)

    def adjacency_matrix(self):
        n_node = self.dgl_graph.num_nodes()
        if n_node < 2:
            adj = torch.tensor([[0]])
        else:
            indices = torch.stack(self.dgl_graph.all_edges())
            values = self.dgl_graph.edata['w'].squeeze()
            adj = torch.sparse_coo_tensor(indices, values, size= (n_node, n_node)).to_dense()

        return adj

    def to(self, device: str= 'cpu'):
        self.dgl_graph = self.dgl_graph.to(device)
        return self
    
    def reverse(self):
        self.dgl_graph = dgl.reverse(self.dgl_graph, copy_ndata= True, copy_edata= True)
        return self

    def set_all_positional_encoding(self, d_pos: int= None, n: int= None):
        """
        if n is not None, encoding as all nodes have n children.
        """
        d_pos = d_pos if d_pos else self.max_depth * self.max_degree
        self.dgl_graph.ndata['pos'] = torch.zeros(self.dgl_graph.num_nodes(), d_pos)
        for nid in self.dgl_graph.nodes()[1:]:    
            parent = self.dgl_graph.successors(nid)
            if len(parent) > 0:
                parent = parent[0]
            else:
                continue
            children = self.dgl_graph.predecessors(parent).tolist()

            n = n if n else len(children)
            assert n >= len(children)
            positional_encoding = [0.0 for _ in range(n)]
            positional_encoding[children.index(nid)] = 1.0
            positional_encoding += self.dgl_graph.ndata['pos'][parent].tolist()
            
            self.dgl_graph.ndata['pos'][nid] = torch.tensor(positional_encoding)[:d_pos]

    def set_positional_encoding(self, nid: int, num_sibling: int= None, d_pos: int= None):
        d_pos = d_pos if d_pos else self.max_depth * self.max_degree
        
        parents = self.dgl_graph.successors(nid)
        if len(parents) == 0:
            self.dgl_graph.ndata['pos'] = torch.zeros(self.dgl_graph.num_nodes(), d_pos)
            positional_encoding = [0.0] * d_pos
        else:
            parent = parents[0]
            sibling = self.dgl_graph.predecessors(parent).tolist()
            num_sibling = num_sibling if num_sibling is not None else len(sibling)
            assert num_sibling >= len(sibling)

            positional_encoding = [0.0 for _ in range(num_sibling)]
            positional_encoding[sibling.index(nid)] = 1.0
            positional_encoding += self.dgl_graph.ndata['pos'][parent].tolist()
            
        self.dgl_graph.ndata['pos'][nid] = torch.tensor(positional_encoding)[:d_pos]

    def width(self, level: int):
        return (self.dgl_graph.ndata['level'] == level).sum()


class BatchedFragmentTree:
    def __init__(self, tree_list, max_depth: int= None, max_degree: int= None):
        graph_list = []
        depth_list, degree_list = zip(*[(tree.max_depth, tree.max_degree) for tree in tree_list])
        if (max_depth is None) | (max_degree is None):
            self.max_depth = max(depth_list)
            self.max_degree = max(degree_list)
        else:
            if (max_depth < max(depth_list)) | (max_degree < max(degree_list)):
                print(f'[WARNING] max depth:{max_depth} < {max(depth_list)} or max degree:{max_degree} < {max(degree_list)}', flush= True)
            self.max_depth = max_depth
            self.max_degree = max_degree

        for tree in tree_list:
            tree.set_all_positional_encoding(d_pos= self.max_depth * self.max_degree)
            graph_list.append(tree.dgl_graph)
        self.batch_dgl_graph = dgl.batch(graph_list)

    def get_ndata(self, key: str, node_ids: list= None, pad_value: int= 0):
        graph_list = dgl.unbatch(self.batch_dgl_graph)
        ndatas = []
        max_nodes_num = max([graph.num_nodes() for graph in graph_list])
        for i, graph in enumerate(graph_list):
            if node_ids:
                node_id = node_ids[i] if i < len(node_ids) else node_ids[0]
                states = graph.ndata[key][node_id]
            else:
                states = graph.ndata[key]
                node_num, state_num = states.size()
                if len(states) < max_nodes_num:
                    padding = states.new_full((max_nodes_num - node_num, state_num), pad_value)
                    states = torch.cat((states, padding), dim=0)
            ndatas.append(states)
        return torch.stack(ndatas)
    
    def get_edata(self, key: str= 'w', edge_ids: list= None, pad_value: int= 0):
        graph_list = dgl.unbatch(self.batch_dgl_graph)
        edatas = []
        max_edges_num = max([graph.num_edges() for graph in graph_list])
        for i, graph in enumerate(graph_list):
            if edge_ids:
                edge_id = edge_ids[i] if i < len(edge_ids) else edge_ids[0]
                states = graph.edata[key][edge_id]
            else:
                states = graph.edata[key]
                edge_num, state_num = states.size()
                if len(states) < max_edges_num:
                    padding = states.new_full((max_edges_num - edge_num, state_num), pad_value)
                    states = torch.cat((states, padding), dim=0)
            edatas.append(states)
        return torch.stack(edatas)

    def get_tree_list(self):
        graph_list = dgl.unbatch(self.batch_dgl_graph)
        return [FragmentTree(dgl_graph= graph) for graph in graph_list]

    def to(self, device: str= 'cpu'):
        self.batch_dgl_graph = self.batch_dgl_graph.to(device)
        return self
    
    def reverse(self):
        reverse_graph_list = [dgl.reverse(g, copy_ndata= True, copy_edata= True) for g in dgl.unbatch(self.batch_dgl_graph)]
        self.batch_dgl_graph = dgl.batch(reverse_graph_list)
        
        return self
  

def make_tree(frag_indices: list, ecfps: torch.Tensor, bond_types: list, bondMapNums: list, d_pos: int= None) -> FragmentTree:
    """
    frag_indices: a list of fragments indices
    ecfps: ecfps of fragments, shape= (len(frag_indices), n_bits)
    bond_types: a list of bondtype (1: single, 2: double, 3: triple)
    bondMapNums: a list of connection order lists. 
                 ex. [[1], [1, 2], [2]] -> first connect frag0 and 1, next frag1 and 2.
    d_pos: dimension of positional encoding
    """
    if type(ecfps) == list:
        ecfps = torch.tensor(ecfps).float()

    tree = FragmentTree()
    tree.add_node(parent_id= None, feature= ecfps[0], fid= frag_indices[0], bondtype= 0)

    stack = [0]
    node_ids = [0] * len(frag_indices)
    while max(map(len, bondMapNums)) > 0:
        if stack:
            parent = stack[-1]
            pid = node_ids[parent]
            if bondMapNums[parent]:
                b = bondMapNums[parent].pop(0)
            else:
                stack.pop(-1)
                continue
        else:
            # accept partial trees
            idx = [i for i in range(len(frag_indices)) if len(bondMapNums[i]) > 0][0]
            stack.append(idx)
            add_node_id = tree.add_node(parent_id= None, feature= ecfps[idx], fid= frag_indices[idx], bondtype= 0)
            node_ids[idx] = add_node_id
            continue

        child_list = [b in mapnums for mapnums in bondMapNums]
        if np.any(child_list):
            c = child_list.index(True)
            add_node_id = tree.add_node(parent_id= pid, feature= ecfps[c], fid= frag_indices[c], bondtype= bond_types[b-1])
            node_ids[c] = add_node_id
            stack.append(c)

    if d_pos:
        tree.set_all_positional_encoding(d_pos)

    return tree


def get_tree_features(frag_indices: list, ecfps: torch.Tensor, bond_types: list, bondMapNums: list, 
                      max_depth: int= None, max_degree: int= None, free_n: bool= False):
    tree = make_tree(frag_indices, ecfps, bond_types, bondMapNums)

    max_depth = max_depth if max_depth else tree.max_depth
    max_degree = max_degree if max_degree else tree.max_degree

    if (max_depth < tree.max_depth) | (max_degree < tree.max_degree):
        print(f'[WARNING] max depth:{max_depth} < {tree.max_depth} or max degree:{max_degree} < {tree.max_degree}', flush= True)
    
    n = None if free_n else max_degree
    tree.set_all_positional_encoding(d_pos= max_depth * max_degree, n= n)
    fids = tree.dgl_graph.ndata['fid'].squeeze(-1)
    positions = tree.dgl_graph.ndata['pos']
    features = tree.dgl_graph.ndata['x']

    return fids, features, positions


def get_pad_features(tree_list, key: str, max_nodes_num: int):
    ndatas = []
    for tree in tree_list:
        states = tree.dgl_graph.ndata[key]
        node_num, state_num = states.size()
        if len(states) < max_nodes_num:
            padding = states.new_full((max_nodes_num - node_num, state_num), 0)
            states = torch.cat((states, padding), dim=0)
        ndatas.append(states)
    return torch.stack(ndatas)


if __name__ == '__main__':
    frag_indice = [0, 1, 2, 3]
    ecfps = torch.ones(4, 12).float()
    bondtypes = [1, 1, 1]
    bondMapNum = [[1], [1, 2], [2, 3]]

    tree = make_tree(frag_indice, ecfps, bondtypes, bondMapNum, d_pos= 16)
    print(tree)