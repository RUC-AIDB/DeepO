import torch
from torch import nn
import numpy as np

class LabelEncoder(object):
    
    """Encoder for objects with tree structure.

    Args:
        value_fn: The function used to extract features from nodes.
            Should either return one-dimensional tensors,
            lists, or tuples.
        children_fn: The function used to get the children of nodes.
    """

    def __init__(self, value_fn, children_fn):
        super(LabelEncoder, self).__init__()
        self.value_fn = value_fn
        self.children_fn = children_fn


    def encode(self, tree):
        """Encodes a tree-like object.

        Args:
            tree: A tree-like object.

        Returns:
            (Tensor, LongTensor): A pair of two tensors,
            one for the values of the tree and one for the arities.

            The returned tensor of values is of shape `(N, features_size)`,
            where `N` is the number of elements in the tree and `features_size`
            is the number of features.

            The returns tensor of arities is of shape `(N,)`,
            where `N` is the number of elements in the tree.

            The values and arities appear in right-first post-order.
        """

        children = self.children_fn(tree)
        # print(children)
        n = len(children)
        value = self.value_fn(tree)
        # if type(value) is list or type(value) is tuple:
        value = torch.from_numpy(np.array(value))

        if children:
            lower_values, lower_arities = \
                zip(* [self.encode(c) for c in reversed(children)])

            lower_values = list(lower_values)
            lower_arities = list(lower_arities)
        else:
            lower_values = []
            lower_arities = []
        lower_values.append(value.unsqueeze(0))
        # lower_values.append(value)
        lower_arities.append(torch.LongTensor([len(children)]))
        return torch.cat(lower_values), torch.cat(lower_arities)


    def encode_batch(self, trees, batch_first=False, ignore_value=None):
        """Encodes a sequence of tree-like objects.

        Args:
            trees: A sequence of tree-like objects.
            batch_first: If ``True``, the values are returned with the
                batch dimension first. Otherwise, the temporal dimension is
                returned first.
                Default: ``False``
            ignore_value: The features used to pad the tensor of features.
                Can either be a one dimensional Tensor, a list or a tuple.
                Default: Zeros.

        Returns:
            (Tensor, LongTensor): A pair of two tensors,
            one for the values of the trees and one for the arities.

            The returned tensor of values is of shape
            `(N, batch_size, features_size)`, where
            `N` is the largest number of elements in the trees,
            `batch_size` is the number of trees, and
            `features_size` is the number of features.

            The returns tensor of arities is of shape `(N, batch_size)`, where
            `N` is the largest number of elements in the trees, and
            `batch_size` is the number of trees.

            The values are padded by `ignore_value` (by default zeros), and
            the arities are padded by ``-1``.

            The values and arities appear in post-order.
        """

        if type(ignore_value) is list or type(ignore_value) is tuple:
            ignore_value = torch.FloatTensor(ignore_value)

        batch_dim = 0 if batch_first else 1
        all_values = []
        all_arities = []
        max_size = 0
        for tree in trees:
            values, arities = self.encode(tree)
            all_values.append(values)
            all_arities.append(arities)
            max_size = max(max_size, values.size(0))

        def pad_values(tensor):
            dims = list(tensor.size())
            dims[0] = max_size - dims[0]
            if ignore_value is not None:
                padding = ignore_value.unsqueeze(0).expand(dims)
            else:
                padding = tensor.new(* dims).fill_(0)
            return torch.cat([tensor, padding])

        def pad_arities(tensor):
            pad_size = max_size - tensor.size(0)
            padding = tensor.new(pad_size).fill_(-1)
            return torch.cat([tensor, padding])

        all_values = [pad_values(v) for v in all_values]
        all_arities = [pad_arities(a) for a in all_arities]

        all_values = torch.stack(all_values, dim=batch_dim)
        all_arities = torch.stack(all_arities, dim=batch_dim)

        return all_values, all_arities

class TreeEncoder(object):
    """Encoder for objects with tree structure.

    Args:
        value_fn: The function used to extract features from nodes.
            Should either return one-dimensional tensors,
            lists, or tuples.
        children_fn: The function used to get the children of nodes.
    """

    def __init__(self, value_fn, children_fn):
        super(TreeEncoder, self).__init__()
        self.value_fn = value_fn
        self.children_fn = children_fn

    def encode(self, tree):
        """Encodes a tree-like object.

        Args:
            tree: A tree-like object.

        Returns:
            (Tensor, LongTensor): A pair of two tensors,
            one for the values of the tree and one for the arities.

            The returned tensor of values is of shape `(N, features_size)`,
            where `N` is the number of elements in the tree and `features_size`
            is the number of features.

            The returns tensor of arities is of shape `(N,)`,
            where `N` is the number of elements in the tree.

            The values and arities appear in right-first post-order.
        """

        children = self.children_fn(tree)
        n = len(children)
        value = self.value_fn(tree)
        if type(value) is list or type(value) is tuple:
            value = torch.Tensor(value)

        if children:
            lower_values, lower_arities = \
                zip(* [self.encode(c) for c in reversed(children)])

            lower_values = list(lower_values)
            lower_arities = list(lower_arities)
        else:
            lower_values = []
            lower_arities = []
        lower_values.append(value.unsqueeze(0))
        lower_arities.append(torch.LongTensor([len(children)]))
        return torch.cat(lower_values), torch.cat(lower_arities)

    def encode_batch(self, trees, batch_first=False, ignore_value=None):
        """Encodes a sequence of tree-like objects.

        Args:
            trees: A sequence of tree-like objects.
            batch_first: If ``True``, the values are returned with the
                batch dimension first. Otherwise, the temporal dimension is
                returned first.
                Default: ``False``
            ignore_value: The features used to pad the tensor of features.
                Can either be a one dimensional Tensor, a list or a tuple.
                Default: Zeros.

        Returns:
            (Tensor, LongTensor): A pair of two tensors,
            one for the values of the trees and one for the arities.

            The returned tensor of values is of shape
            `(N, batch_size, features_size)`, where
            `N` is the largest number of elements in the trees,
            `batch_size` is the number of trees, and
            `features_size` is the number of features.

            The returns tensor of arities is of shape `(N, batch_size)`, where
            `N` is the largest number of elements in the trees, and
            `batch_size` is the number of trees.

            The values are padded by `ignore_value` (by default zeros), and
            the arities are padded by ``-1``.

            The values and arities appear in post-order.
        """

        if type(ignore_value) is list or type(ignore_value) is tuple:
            ignore_value = torch.FloatTensor(ignore_value)

        batch_dim = 0 if batch_first else 1
        all_values = []
        all_arities = []
        max_size = 0
        for tree in trees:
            values, arities = self.encode(tree)
            all_values.append(values)
            all_arities.append(arities)
            max_size = max(max_size, values.size(0))

        def pad_values(tensor):
            dims = list(tensor.size())
            dims[0] = max_size - dims[0]
            if ignore_value is not None:
                padding = ignore_value.unsqueeze(0).expand(dims)
            else:
                padding = tensor.new(* dims).fill_(0)
            return torch.cat([tensor, padding])

        def pad_arities(tensor):
            pad_size = max_size - tensor.size(0)
            padding = tensor.new(pad_size).fill_(-1)
            return torch.cat([tensor, padding])

        all_values = [pad_values(v) for v in all_values]
        all_arities = [pad_arities(a) for a in all_arities]

        all_values = torch.stack(all_values, dim=batch_dim)
        all_arities = torch.stack(all_arities, dim=batch_dim)

        return all_values, all_arities

class TreeNet(nn.Module):
    """Class for recursive neural networks with n-ary tree structure.

    The class supports batch processing of tree-like objects with
    bounded branching factor.
    The class is intended as a base class for recursive neural networks.

    Given a `unit` network for processing single nodes (see note below),
    the TreeNet class returns a network capable of processing (properly
    encoded) trees.

    Note:
        The `unit` network specifies what should be done for each node of
        the input trees. It receives as input three parameters:
            - inputs: A Tensor containing the input features of
                the current nodes. Of shape `(batch_size, input_size)`.
            - children: A list, of size `branching_factor`, of Tensors
                containing the output features of the children of the
                current nodes.
                Each Tensor has the shape `(batch_size, output_size)`.
                If a node has less arity than the `branching_factor`,
                the features corresponding to children absent from the
                node are guaranteed to have magnitude zero.
            - arities: A LongTensor containing the arities of the nodes.
                Of shape `(batch_size,)`.
        The `unit` network should return the output features for the current
        nodes, which should be of shape `(batch_size, output_size)`.

    Args:
        output_size (int): Number of features output by the `unit` network.
        branching_factor (int): Largest branching factor of input trees.
        unit (torch.nn.Module): Network used for processing nodes.

    See Also:
        See the `treenet.encoder` module for how to encode trees and batches
        of trees.

    References:
        Bowman, S. R., Gauthier, J., Rastogi, A., Gupta, R.,
        Manning, C. D., & Potts, C. (2016).
        A Fast Unified Model for Parsing and Sentence Understanding.
    """

    def __init__(self, output_size, branching_factor=2, unit=None):
        super(TreeNet, self).__init__()
        self.output_size = output_size
        self.branching_factor = branching_factor
        if unit is not None:
            self.unit = unit

    def forward(self, inputs, arities, batch_first=False):
        """Feed the network with encoded tree-like objects.

        Args:
            inputs (Tensor): The features.
                Should be of shape `(time, batch_size, input_size)`.
            arities (LongTensor): The arities of the nodes.
                Should be of shape `(time, batch_size)`.
            batch_first (bool): If ``True``, then `inputs` and `arities`
                are expected to have the batch dimension first.
        Returns:
            Tensor: The output features,
                of shape `(batch_size, output_size)`.
        """

        if batch_first:
            inputs = inputs.permute(1, 0, 2)
            arities = arities.permute(1, 0)

        # Time size.
        T = inputs.size(0)

        # Batch size.
        B = inputs.size(1)

        # 0, 1 .. B - 1. Used for indexing.
        k = arities.new(range(B))

        # Memory will contain the state of every node.
        memory = inputs.new_zeros(T, B, self.output_size)

        # The stack maintains pointers to the memory for unmerged subtrees.
        # It contains extra entries, to avoid out of bounds accesses.
        stack = arities.new_zeros(B, T + self.branching_factor)

        # Points to the head of the stack.
        # Starts at the given index in order to avoid out of bounds reads.
        stack_pointer = arities.new_full((B,), self.branching_factor - 1)

        for t in range(T):
            arity = arities[t]
            current = inputs[t]

            entries = []
            for i in range(self.branching_factor):
                entry = memory[stack[k, stack_pointer - i], k]
                mask = entry.new_empty(B)
                mask.copy_(arity > i)
                mask = mask.unsqueeze(1).expand(entry.size())
                entries.append(entry * mask)

            # Obtain the state for the node.
            new_entry = self.unit(current, entries, arity)

            # If multiple entries are returned, each entry must be
            # appropriately masked.
            # if type(new_entry) is list or type(new_entry) is tuple:
            #     for i, entry in enumerate(new_entry):
            #         factors = entry.new_empty(B)
            #         factors.copy_(arity == i)
            #         factors = factors.unsqueeze(1).expand(entry.size())
            #         memory[t] = memory[t] + (entry * factors)
            # else:
            #     memory[t] = new_entry
            memory[t] = new_entry


            # Update the stack pointer.
            stack_pointer.add_(-torch.abs(arity) + 1)

            # Ensure that the top of the stack is untouched if the arity is the
            # special value -1.
            ignore = (arity == -1).long()
            stack[k, stack_pointer] *= ignore
            stack[k, stack_pointer] += t * ((ignore + 1) % 2)

        # Return the content of the memory location
        # pointed by the top of the stack.
        # print(memory)
        # print(memory[stack[k, stack_pointer], k])
        return memory[stack[k, stack_pointer], k],memory
        
class MyTreeLSTMUnit(nn.Module):
    def __init__(self, input_size, memory_size, branching_factor):
        super(MyTreeLSTMUnit, self).__init__()

        self.input_size = input_size
        self.memory_size = memory_size
        self.branching_factor = branching_factor

        self.wi_net = nn.Linear(self.input_size, self.memory_size)
        self.wo_net = nn.Linear(self.input_size, self.memory_size)
        self.wu_net = nn.Linear(self.input_size, self.memory_size)
        self.wf_net = nn.Linear(self.input_size, self.memory_size)

        self.ui_nets = []
        self.uo_nets = []
        self.uu_nets = []
        self.uf_nets = []

        for i in range(branching_factor):
            ui = nn.Linear(self.memory_size, self.memory_size, bias=False)
            self.add_module("ui_net_{}".format(i), ui)
            self.ui_nets.append(ui)

            uo = nn.Linear(self.memory_size, self.memory_size, bias=False)
            self.add_module("uo_net_{}".format(i), uo)
            self.uo_nets.append(uo)

            uu = nn.Linear(self.memory_size, self.memory_size, bias=False)
            self.add_module("uu_net_{}".format(i), uu)
            self.uu_nets.append(uu)
            
            uf = nn.Linear(self.memory_size, self.memory_size, bias=False)
            self.add_module("uf_net_{}".format(i), uf)
            self.uf_nets.append(uf)

        for p in self.parameters():
            nn.init.normal_(p)

    def forward(self, inputs, children, arities):

        i = self.wi_net(inputs)
        o = self.wo_net(inputs)
        u = self.wu_net(inputs)

        f_base = self.wf_net(inputs)
        fc_sum = inputs.new_zeros(self.memory_size)
        for k, child in enumerate(children):
            child_h, child_c = torch.chunk(child, 2, dim=1)
            i.add_(self.ui_nets[k](child_h))
            o.add_(self.uo_nets[k](child_h))
            u.add_(self.uu_nets[k](child_h))
            f = f_base
            f = f.add(self.uf_nets[k](child_h))
            fc_sum.add(torch.sigmoid(f) * child_c)

        
        c = torch.sigmoid(i) * torch.tanh(u) + fc_sum
        h = torch.sigmoid(o) * torch.tanh(c)
        return torch.cat([h, c], dim=1)

class TreeLSTM(TreeNet):
    """Tree-LSTM network.
    Args:
        input_size (int): Number of input features.
        memory_size (int): Number of output features.
        branch_factor (int): Maximal branching factor for input trees.
    """
    def __init__(self, input_size, memory_size, branching_factor):
        # unit = TreeLSTMUnit(input_size, memory_size, branching_factor)
        unit = MyTreeLSTMUnit(input_size,memory_size,branching_factor)
        super(TreeLSTM, self).__init__(memory_size * 2, branching_factor, unit)

    def forward(self, *args, **kwargs):
        hc,_all = super(TreeLSTM, self).forward(*args, **kwargs)
        h, c = torch.chunk(hc, 2, dim=1)
        _h,_c = torch.chunk(_all,2,dim=2)
        return h,_h

class Tree(torch.nn.Module):
    def __init__(self,in_features,hidden_dim,branching_factor):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.in_features = in_features
        self.branching_factor = branching_factor
        self.lstm = TreeLSTM(in_features,hidden_dim,branching_factor)
        self.hidden2scalar = torch.nn.Sequential(torch.nn.Linear(hidden_dim,128),torch.nn.ReLU(),\
                                                #torch.nn.Linear(128,128),torch.nn.ReLU(),\
                                                torch.nn.Linear(128,64),torch.nn.ReLU(),\
                                                torch.nn.Linear(64,1),torch.nn.ReLU())
        # self.hidden2scalar = torch.nn.Sequential(torch.nn.Linear(hidden_dim,1),torch.nn.ReLU())
    def forward(self,inputs,arities):
        h,h_step = self.lstm(inputs,arities)
        output = self.hidden2scalar(h_step)
        return output

def convert_tree(father):
    t = tuple(father['features'])
    tmp = [t]
    tmp = tuple(tmp)
    c = []
    for child in father['children']:
        c.append(tuple(convert_tree(child)))
    tmp += (c,)
    return tmp


net = Tree(79, 256, branching_factor=2).cuda()
net.load_state_dict(torch.load("../model/treelstm_model"))

operators = ['Merge Join', 'Hash', 'Index Only Scan using title_pkey on title t', 'Sort','Seq Scan',\
              'Index Scan using title_pkey on title t', 'Materialize', 'Nested Loop', 'Hash Join']

import pickle
with open("./final/test_64_cat_tree.pkl","rb") as f:
# with open("./final/job-light_64_cat_tree.pkl","rb") as f:
    test_trees = pickle.load(f)
test_converted_trees = []
for each in test_trees:
    test_converted_trees.append(convert_tree(each))
print(len(test_converted_trees))

# loss_fucti
ec = TreeEncoder(lambda x: x[0],lambda x: x[1])
lec = LabelEncoder(lambda x: x['labels'],lambda x: x['children'])
pgec = LabelEncoder(lambda x:x['pg'],lambda x:x['children'])
operator = []
final_result = []
all_result = []
label_final_result = []
label_all_result = []
pg_all = []
pg_final = []
# batch_size = len(test_converted_trees)
batch_size = 3
num_of_batch = 1
# num_of_batch = 0
with torch.no_grad():
    net.eval()
    for cnt in range(num_of_batch):
        inputs, arities = ec.encode_batch(test_converted_trees[cnt*batch_size:(cnt+1)*batch_size])
        label_inputs, label_arities = lec.encode_batch(test_trees[cnt*batch_size:(cnt+1)*batch_size])
        pg_inputs, pg_arities = pgec.encode_batch(test_trees[cnt*batch_size:(cnt+1)*batch_size])
        label_inputs = label_inputs.view(-1,batch_size,1).float()
        inputs, arities = inputs.cuda(), arities.cuda()
        label_inputs,label_arities = label_inputs.cuda(),label_arities.cuda()

        output = net.forward(inputs, arities)
#         loss = loss_function(output, label_inputs)
        for i in range(arities.size()[1]):
#     print(i)
            for j in range(arities.size()[0]):
                if(arities[j][i]!=-1):
                    operator.append(operators[list(inputs[j][i].cpu().numpy()).index(1)])
                    all_result.append(output[j][i].cpu().numpy())
                    label_all_result.append(label_inputs[j][i].cpu().numpy())
                    pg_all.append(pg_inputs[j][i])
                else:
                    final_result.append(output[j-1][i].cpu().numpy())
                    pg_final.append(pg_inputs[j-1][i])
                    label_final_result.append(label_inputs[j-1][i].cpu().numpy())
                    break
                if(j==arities.size()[0]-1):
                    final_result.append(output[j][i].cpu().numpy())
                    pg_final.append(pg_inputs[j][i])
                    label_final_result.append(label_inputs[j][i].cpu().numpy())