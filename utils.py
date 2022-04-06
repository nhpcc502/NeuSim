import os
import time
import re
import ast
import torch
import pandas as pd

from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

MAX_TGT_LEN = 20

OPERATORS = {
    'Invert': '~',
    'Add': '+',
    'Sub': '-',
    'BitAnd': '&',
    'USub': '-',
    'BitOr': '|',
    'Mult': '*',
    'BitXor': '^',
    'UAdd': '+',
    'Pow': '**'
}

class ExprVisit(ast.NodeTransformer):
    '''
    Parsing an expression and generating the AST by the post-order traversal.
    For each expression, its variables must be a single character.
    '''
    def __init__(self):
        self.node_list = []
        self.edge_list = []
        self.subtree_memo = []

    def merge_node(self, node):
        '''Determine whether the current node can be merged'''
        same_node = None
        if self.subtree_memo is None:
            return same_node
        cur_node_type = 'binop' if hasattr(node, 'left') else 'unaryop'
        for p_node in self.subtree_memo:
            cur_p_node_type = 'binop' if hasattr(p_node, 'left') else 'unaryop'
            if cur_p_node_type != cur_node_type:
                continue
            elif cur_p_node_type == cur_node_type == 'binop':
                if ast.dump(p_node.op) == ast.dump(node.op) \
                    and ast.dump(p_node.left) == ast.dump(node.left) \
                    and ast.dump(p_node.right) == ast.dump(node.right):
                    same_node = ast.dump(p_node.op) + str(p_node.col_offset)
            elif cur_p_node_type == cur_node_type == 'unaryop':
                if ast.dump(p_node.op) == ast.dump(node.op) \
                    and ast.dump(p_node.operand) == ast.dump(node.operand):
                    same_node = ast.dump(p_node.op) + str(p_node.col_offset)

        return same_node

    def visit_BinOp(self, node):
        '''
        Scaning binary operators, such as +, -, *, &, |, ^
        '''
        self.generic_visit(node)
        # node.col_offset is unique identifer
        node_str = ast.dump(node.op) + str(node.col_offset)

        # Merge same subtrees or leaves
        same_node = self.merge_node(node)
        if same_node == None:
            self.subtree_memo.append(node)
            self.node_list.append(node_str)
        else:
            for idx in range(len(self.edge_list) - 1, -1, -1):
                if node_str == self.edge_list[idx][0] or \
                    node_str == self.edge_list[idx][1]:
                    del self.edge_list[idx]
            node_str = same_node

        # If this node has parent, then create a edge between the node and its parent
        if hasattr(node.parent, 'op'):
            node_parent_str = ast.dump(node.parent.op) + str(node.parent.col_offset)
            self.edge_list.append([node_str, node_parent_str])

        return node

    def visit_UnaryOp(self, node):
        '''
        Scaning unary operators, such as - and ~
        '''
        self.generic_visit(node)
        node_str = ast.dump(node.op) + str(node.col_offset)
        
        # Merge same subtrees or leaves
        same_node = self.merge_node(node)
        if same_node == None:
            self.subtree_memo.append(node)
            self.node_list.append(node_str)
        else:
            for idx in range(len(self.edge_list) - 1, -1, -1):
                if node_str == self.edge_list[idx][0] \
                    or node_str == self.edge_list[idx][1]:
                    del self.edge_list[idx]
            node_str = same_node

        # If this node has parent, then create a edge between the node and its parent
        if hasattr(node.parent, 'op'):
            node_parent_str = ast.dump(node.parent.op) + str(node.parent.col_offset)
            self.edge_list.append([node_str, node_parent_str])

        return node

    def visit_Name(self, node):
        '''
        Scaning variables
        '''
        self.generic_visit(node)
        # node.col_offset will allocate a unique ID to each node
        node_str = node.id

        if node_str not in self.node_list:
            self.node_list.append(node_str)
        # If this node has parent, then create a edge between the node and its parent
        if hasattr(node.parent, 'op'):
            node_parent_str = ast.dump(node.parent.op) + str(node.parent.col_offset)
            self.edge_list.append([node_str, node_parent_str])

        return node

    # Use visit_Constant when python version >= 3.8
    # def visit_Constant(self, node):
    def visit_Num(self, node):
        '''
        Scaning numbers
        '''
        self.generic_visit(node)
        node_str = str(node.n)

        if node_str not in self.node_list:
            self.node_list.append(node_str)
        node_parent_str = ast.dump(node.parent.op) + str(node.parent.col_offset)
        self.edge_list.append([node_str, node_parent_str])

        return node

    def get_result(self):
        return self.node_list, self.edge_list


def expr2graph(expr):
    '''
    Convert a expression to a MSAT graph.

    Parameters:
        expr: A string-type expression.

    Return:
        node_list:  List for nodes in graph.
        edge_list:  List for edges in graph.
    '''
    ast_obj = ast.parse(expr)
    for node in ast.walk(ast_obj):
        for child in ast.iter_child_nodes(node):
            child.parent = node
    
    vistor = ExprVisit()
    vistor.visit(ast_obj)
    node_list, edge_list = vistor.get_result()

    return node_list, edge_list


def repl(integer):
    integer = integer.group(0)
    if len(integer) == 1: 
        return integer

    expr, length = '', len(integer)

    for i in range(length):
        if integer[i] == '0': continue
        if i == length - 1: expr += '+' + integer[i]
        elif i == length - 2: expr += ('' if length == 2 else '+') + integer[i] + '*10'
        else: expr += ('+' if i else '') + integer[i] + f'*10**{length-1-i}'

    return  '(' + expr + ')'
def num_decompose(expr):
    return re.sub(r'\d+\.?\d*', repl, expr)


class GraphExprDataset(InMemoryDataset):
    '''
    Base class of our dataset.
    '''
    def __init__(self, root, dataset):
        self.dataset = dataset
        self.qst_vocab = {}
        self.ans_vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2}
        self.qst_vocab['10'] = len(self.qst_vocab)
        self.qst_vocab['**'] = len(self.qst_vocab)
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return f'dataset/raw/{self.dataset}.csv'
    
    @property
    def processed_file_names(self):
        return f'{self.dataset}.pt'
    
    def download(self):
        pass

    def process(self):
        data_list = []

        df = pd.read_csv(self.raw_file_names, header=None, nrows=None)
        
        max_tgt_len = float('-inf')
        
        # one sample per row
        for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
            # row[0] is source expression, which will be transformed into a graph
            raw_qst, raw_ans = str(row[0]), str(row[1])
            expr = num_decompose(raw_qst).replace('+-', '-').replace('--', '+')
            
            for c in expr:
                # qst_vocab因为会转换成树，不需要括号
                if c != '(' and c != ')' and c not in self.qst_vocab:
                    self.qst_vocab[c] = len(self.qst_vocab)
            for c in raw_ans:
                if c not in self.ans_vocab:
                    self.ans_vocab[c] = len(self.ans_vocab)

            x, edge_index = self._generate_graph(expr)
            # print(expr)

            # raw_ans is target, which will be represented as a one-hot vector
            # if len(raw_ans) <= MAX_TGT_LEN:
            y = [self.ans_vocab[c] for c in raw_ans]
            y.insert(0, self.ans_vocab['<sos>'])
            y.append(self.ans_vocab['<eos>'])

            max_tgt_len = max(max_tgt_len, len(y))

            y = torch.tensor(y, dtype=torch.long)

            # Fed the graph and the label into Data
            data = Data(x=x, edge_index=edge_index, y=y, src=raw_qst)
            # data = Data(x=x_s, edge_index=edge_index_s)
            data_list.append(data)

        # Pad the target
        for data in data_list:
            padding = torch.zeros(max_tgt_len - data.y.shape[0], dtype=torch.long)
            # padding = torch.tensor([self.ans_vocab['<pad>']] * (max_tgt_len - data.y.shape[0]), dtype=torch.long)
            data.y = torch.cat((data.y, padding), dim=0)
            padding = torch.zeros((data.x.shape[0], len(self.qst_vocab)-data.x.shape[1]), dtype=torch.float)
            data.x = torch.cat((data.x, padding), dim=1)

        self.max_tgt_len = max_tgt_len
        # exit(0)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def _generate_graph(self, expr):
        try:
            node_list, edge_list = expr2graph(expr)
        except:
            raise ValueError(expr)

        node_feature = []

        for node in node_list:
            tag = node.split('()')[0]
            if tag in OPERATORS: 
                tag = OPERATORS[tag]

            feature = [0] * len(self.qst_vocab)
            feature[self.qst_vocab[tag]] = 1
            
            node_feature.append(feature)

        COO_edge_idx = [[], []]
        for edge in edge_list:
            s_node, e_node = node_list.index(edge[0]), node_list.index(edge[1])
            COO_edge_idx[0].append(s_node), COO_edge_idx[1].append(e_node)

        x = torch.tensor(node_feature, dtype=torch.float)
        edge_index = torch.tensor(COO_edge_idx, dtype=torch.long)

        return x, edge_index
