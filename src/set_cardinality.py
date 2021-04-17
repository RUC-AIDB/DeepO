""" generate cardinality injection queries
    Input: one query plan
    Output: a sql query
    process: embedding leaf operators of query plan
             generate embedded tree for TreeLSTM
             TreeLSTM evaluate the tree
             check all operators for subplan with qerror larger than threshold
             generate set qeries
"""

import numpy as np
import argparse
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import csv
import torch
from torch import nn
from sklearn import preprocessing

def extract_plan(line):
    data = line.replace("->","").lstrip().split("  ")[-1].split(" ")
    start_cost = data[0].split("..")[0].replace("(cost=","")
    end_cost = data[0].split("..")[1]
    rows = data[1].replace("rows=","")
    width = data[2].replace("width=","").replace(")","")
    a_start_cost = data[4].split("..")[0].replace("time=","")
    a_end_cost = data[4].split("..")[1]
    a_rows = data[5].replace("rows=","") 
    return float(start_cost),float(end_cost),float(rows),float(width),float(a_start_cost),float(a_end_cost),float(a_rows)

def get_column_statistics(path="/home/sunluming/demo/data/column_min_max_vals.csv"):
    with open(path, 'r') as f:
        data_raw = list(list(rec) for rec in csv.reader(f, delimiter=','))
        column_min_max_vals = {}
        for i, row in enumerate(data_raw[1:]):
            column_min_max_vals[row[0]] = [float(row[1]), float(row[2])]
    return column_min_max_vals

def normalize_data(val,column_name,column_min_max_vals):
    min_val = column_min_max_vals[column_name][0]
    max_val = column_min_max_vals[column_name][1]
    val = float(val)
    if(val>max_val):
        val = max_val
    elif(val<min_val):
        val = min_val
    val = float(val)
    val_norm = (val - min_val) / (max_val - min_val)
    return val_norm

def is_not_number(s):
    try:
        float(s)
        return False
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return False
    except (TypeError, ValueError):
        pass
    return True

def get_data_and_label(column_min_max_vals,plan_path):
    sentences = []
    rows = []
    pg = []
    with open(plan_path,'r') as f:
        plan = f.readlines()
    for i in range(len(plan)-2):
        if("Seq Scan" in plan[i]):
            _start_cost,_end_cost,_rows,_width,_a_start_cost_,_a_end_cost,_a_rows = extract_plan(plan[i])
            if(len(plan[i].strip().split("  "))==2):
                _sentence = " ".join(plan[i].strip().split("  ")[0].split(" ")[:-1]) + " "
                table = plan[i].strip().split("  ")[0].split(" ")[4]
            else:
                _sentence = " ".join(plan[i].strip().split("  ")[1].split(" ")[:-1]) + " "
                table = plan[i].strip().split("  ")[1].split(" ")[4]
            if("actual" not in plan[i+1] and "Plan" not in plan[i+1]):
                _sentence += plan[i+1].strip()
            else:
                _sentence += table
                _sentence = _sentence + ' ' + _sentence
            _sentence = _sentence.replace(": "," ").replace("(","").replace(")","").replace("'","").replace("::bpchar","")\
                .replace("[]","").replace(","," ").replace("\\","").replace("::numeric","").replace("  "," ")\
                .replace("Seq Scan on ","").strip()
            sentence = []
            ll = _sentence.split(" ")
            for cnt in range(len(ll)):                 
                if is_not_number(ll[cnt]):
                    sentence.append(ll[cnt])
                else:
                    try:
                        sentence.append(normalize_data(ll[cnt],table+'.'+str(ll[cnt-2]),column_min_max_vals))
                    except:
                        pass
            sentences.append(tuple(sentence))
            rows.append(_a_rows)
            pg.append(_rows)
    return sentences,rows,pg

def prepare_data_and_label(sentences,rows,vocab_dict,vocab_size=25):
    data = []
    label = []
    for sentence,row in zip(sentences,rows):
        _s = []
        for word in sentence:
            if(is_not_number(word)):
                _tmp = np.column_stack((np.array([0]),vocab_dict[word]))
                _tmp = np.reshape(_tmp,(vocab_size+1))
                assert(len(_tmp)==vocab_size+1)
                _s.append(_tmp)
            else:
                _tmp = np.full((vocab_size+1),word)
                assert(len(_tmp)==vocab_size+1)
                _s.append(_tmp)
        data.append(np.array(_s))
        label.append(row)
    return data,label

def get_vocabulary_encoding():
    # leaf node vocabulary
    vocabulary = ['movie_info_idx', 'Filter', 'info_type_id', '>', 'title', 'kind_id', '=', 'production_year', 'movie_keyword', 'keyword_id', 'cast_info', 'person_id', 'AND', 'role_id', 'mk', 't', '<', 'movie_info', 'mi', 'movie_companies', 'mc', 'ci', 'company_id', 'company_type_id', 'mi_idx']
    vocab_size = len(vocabulary)
    _vocabulary = np.array(vocabulary)
    label_encoder = preprocessing.LabelEncoder()
    integer_encoded = label_encoder.fit_transform(_vocabulary)
    encoded = to_categorical(integer_encoded)
    vocab_dict = {}
    for v,e in zip(vocabulary,encoded):
        vocab_dict[v] = np.reshape(np.array(e),(1,vocab_size))
    return vocab_dict

def padding_sentence(raw_sentences,test_data):
    max_len = 0
    for sentence in raw_sentences:
        if(len(sentence) > max_len):
            max_len = len(sentence)
    # print(max_len)
    padded_sentences = pad_sequences(test_data, maxlen=max_len, padding='post',dtype='float32')
    return padded_sentences

def leaf_embedded(plan,model_path,embedded_length=64):
    """embedding leaf node in plan into vector

    Args:
        plan ([path]): path of a plan file
        model_path (str, optional): [description]. Defaults to "/home/sunluming/deepO/Mine_total/final/embedding_model.h5".
        embedded_length (int, optional): [description]. Defaults to 64.
    """
    # base statistics and vocabulary dict for leaf embedding
    column_min_max_vals = get_column_statistics()
    vocab_dict = get_vocabulary_encoding()    
    # extract features from plan
    test_sentences,test_rows,test_pg = get_data_and_label(column_min_max_vals,plan)
    test_data,test_label = prepare_data_and_label(test_sentences,test_rows,vocab_dict,len(vocab_dict))
    padded_sentences = padding_sentence(test_sentences,test_data)

    # load model
    model = load_model(model_path)
    # model.summary()
    intermediate_layer_model = Model(inputs=model.input,
                                    outputs=model.layers[3].output)
    intermediate_output = intermediate_layer_model.predict(padded_sentences)
    return intermediate_output

def extract_operator(line, operators):
    operator = line.replace("->","").lstrip().split("  ")[0]
    if(operator.startswith("Seq Scan")):
        operator = "Seq Scan"
    return operator,operator in operators

def extract_attributes(operator, operators,columns,line,feature_vec,leaf_embedding,i=None):
    operators_count = len(operators) #9
    if(operator in ["Hash","Materialize","Nested Loop"]): 
        pass
    elif(operator=="Merge Join"):
        if("Cond" in line):
            for column in columns:
                if(column in line):
                    feature_vec[columns.index(column)+operators_count] = 1.0
    elif(operator=="Index Only Scan using title_pkey on title t"):
        if("Cond" in line):
            feature_vec[columns.index("t.id")+operators_count] = 1.0
            for column in columns:
                if(column in line):
                    feature_vec[columns.index(column)+operators_count] = 1.0
    elif(operator=="Sort"):
        for column in columns:
            if(column in line):
                feature_vec[columns.index(column)+operators_count] = 1.0          
    elif(operator=='Index Scan using title_pkey on title t'):
        if("Cond" in line):
            feature_vec[columns.index("t.id")+operators_count] = 1.0
            for column in columns:
                if(column in line):
                    feature_vec[columns.index(column)+operators_count] = 1.0
    elif(operator=='Hash Join'):
        if("Cond" in line):
            for column in columns:
                if(column in line):
                    feature_vec[columns.index(column)+operators_count] = 1.0
    elif(operator=='Seq Scan'):
        feature_vec[15:79] = leaf_embedding[i]
    else:
        pass
    # print(feature_vec)

class Node(object):
    def __init__(self, data, parent=None):
        self.data = data
        self.children = []
        self.parent = parent

    def add_child(self, obj):
        self.children.append(obj)
        
    def add_parent(self, obj):
        self.parent = obj
        
    def __str__(self, tabs=0):
        tab_spaces = str.join("", [" " for i in range(tabs)])
        return tab_spaces + "+-- Node: "+ str.join("|", self.data) + "\n"\
                + str.join("\n", [child.__str__(tabs+2) for child in self.children])

def parse_tree(operators,columns,leaf_embedding,plan_path):
    scan_cnt = 0
    max_children = 0
    plan_trees = []
    feature_len = 9+6+7+64
    with open(plan_path,'r') as f:
        lines = f.readlines()
    feature_vec = [0.0]*feature_len
    operator, in_operators = extract_operator(lines[0],operators)
    # print("operator: ",operator)
    if not in_operators:
        operator, in_operators = extract_operator(lines[1],operators)
        start_cost, end_cost, rows, width, a_start_cost, a_end_cost, a_rows = extract_plan(
            lines[1])
        j = 2
    else:
        start_cost, end_cost, rows, width, a_start_cost, a_end_cost, a_rows = extract_plan(
            lines[0])
        j = 1
    feature_vec[feature_len-7:feature_len] = [start_cost,
                                                end_cost, rows, width, a_start_cost, a_end_cost, a_rows]
    feature_vec[operators.index(operator)] = 1.0
    if(operator == "Seq Scan"):
        extract_attributes(operator, operators,columns,lines[j], feature_vec,leaf_embedding, scan_cnt)
        scan_cnt += 1
        # root_tokens = feature_vec
        # current_node = Node(root_tokens)
        # plan_trees.append(current_node)
    else:
        while("actual" not in lines[j] and "Plan" not in lines[j]):
            extract_attributes(operator, operators,columns, lines[j], feature_vec,leaf_embedding)
            j += 1
    root_tokens = feature_vec  # 所有吗
    current_node = Node(root_tokens)
    plan_trees.append(current_node)
    spaces = 0
    node_stack = []
    i = j
    while not lines[i].startswith("Planning time"):
        line = lines[i]
        i += 1
        if line.startswith("Planning time") or line.startswith("Execution time"):
            break
        elif line.strip() == "":
            break
        elif ("->" not in line):
            continue
        else:
            if line.index("->") < spaces:
                while line.index("->") < spaces:
                    current_node, spaces = node_stack.pop()
            if line.index("->") > spaces:
                line_copy = line
                feature_vec = [0.0]*feature_len
                start_cost, end_cost, rows, width, a_start_cost, a_end_cost, a_rows = extract_plan(
                    line_copy)
                feature_vec[feature_len-7:feature_len] = [start_cost,
                                                            end_cost, rows, width, a_start_cost, a_end_cost, a_rows]
                operator, in_operators = extract_operator(line_copy,operators)
                feature_vec[operators.index(operator)] = 1.0
                if(operator == "Seq Scan"):
                    extract_attributes(
                        operator,  operators, columns, line_copy, feature_vec,leaf_embedding, scan_cnt)
                    scan_cnt += 1
                else:
                    j = 0
                    while("actual" not in lines[i+j] and "Plan" not in lines[i+j]):
                        extract_attributes(
                            operator, operators, columns, lines[i+j], feature_vec,leaf_embedding)
                        j += 1
                tokens = feature_vec
                # print("token",tokens)
                new_node = Node(tokens, parent=current_node)
                current_node.add_child(new_node)
                if len(current_node.children) > max_children:
                    max_children = len(current_node.children)
                node_stack.append((current_node, spaces))
                current_node = new_node
                spaces = line.index("->")
            elif line.index("->") == spaces:
                line_copy = line
                feature_vec = [0.0]*feature_len
                start_cost, end_cost, rows, width, a_start_cost, a_end_cost, a_rows = extract_plan(
                    line_copy)
                feature_vec[feature_len-7:feature_len] = [start_cost,
                                                            end_cost, rows, width, a_start_cost, a_end_cost, a_rows]
                operator, in_operators = extract_operator(line_copy,operators)
                feature_vec[operators.index(operator)] = 1.0
                if(operator == "Seq Scan"):
                    extract_attributes(
                        operator, operators, columns, line_copy, feature_vec,leaf_embedding, scan_cnt)
                    scan_cnt += 1
                else:
                    j = 0
                    while("actual" not in lines[i+j] and "Plan" not in lines[i+j]):
                        extract_attributes(
                            operator, operators, columns, lines[i+j], feature_vec,leaf_embedding)
                        j += 1
                tokens = feature_vec
                new_node = Node(tokens, parent=node_stack[-1][0])
                node_stack[-1][0].add_child(new_node)
                if len(node_stack[-1][0].children) > max_children:
                    max_children = len(node_stack[-1][0].children)
                current_node = new_node
                spaces = line.index("->")
    # print("scan count: ",scan_cnt)
    return plan_trees, max_children  # a list of the roots nodes


def p2t(node):
    tree = {}
    tmp = node.data
    operators_count = 9
    columns_count = 6
    scan_features = 64
    # print("node data length: ",len(tmp))
    assert len(tmp) == operators_count + columns_count + 7 +scan_features 
    tree['features']= tmp[:operators_count + columns_count+scan_features]
    if(node.data[-1]!=0):
        tree['labels'] = np.log(node.data[-1])
    else:
        tree['labels'] = np.log(1)
    tree['pg'] = np.log(node.data[-5])
    tree['children'] = []
    for children in node.children:
        tree['children'].append(p2t(children))
    return tree

def tree_embedding(leaf_embedding,plan):
    operators = ['Merge Join', 'Hash', 'Index Only Scan using title_pkey on title t', 'Sort','Seq Scan',\
            'Index Scan using title_pkey on title t', 'Materialize', 'Nested Loop', 'Hash Join']
    columns = ['ci.movie_id', 't.id', 'mi_idx.movie_id', 'mi.movie_id', 'mc.movie_id', 'mk.movie_id']

    root_node, max_children = parse_tree(operators,columns,leaf_embedding,plan)
    embedded_tree = p2t(root_node[0])
    return embedded_tree

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
                                                #  torch.nn.Linear(128,128),torch.nn.ReLU(),\
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

def predict(test_tree,treelstm_model_path):
    net = Tree(79, 256, branching_factor=2)
    
    net.load_state_dict(torch.load(treelstm_model_path,map_location=torch.device('cpu')))

    operators = ['Merge Join', 'Hash', 'Index Only Scan using title_pkey on title t', 'Sort','Seq Scan',\
              'Index Scan using title_pkey on title t', 'Materialize', 'Nested Loop', 'Hash Join']
    test_converted_tree = convert_tree(test_tree)
    # print(test_converted_tree)
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
    batch_size = 1
    num_of_batch = 1
    with torch.no_grad():
        net.eval()
        for cnt in range(num_of_batch):
            inputs, arities = ec.encode_batch([test_converted_tree])
            label_inputs, label_arities = lec.encode_batch([test_tree])
            pg_inputs, pg_arities = pgec.encode_batch([test_tree])
            # print("pg_inputs: ",pg_inputs)
            # print("pg_arities: ",pg_arities)
            label_inputs = label_inputs.view(-1,batch_size,1).float()
            # inputs, arities = inputs.cuda(), arities.cuda()
            # label_inputs,label_arities = label_inputs.cuda(),label_arities.cuda()

            output = net.forward(inputs, arities)
            for i in range(arities.size()[1]):
                for j in range(arities.size()[0]):
                    if(arities[j][i]!=-1):
                        operator.append(operators[list(inputs[j][i].cpu().numpy()).index(1)])
                        all_result.append(output[j][i].cpu().numpy())
                        label_all_result.append(label_inputs[j][i].cpu().numpy())
                        pg_all.append(pg_inputs[j][i].cpu().numpy())
                    else:
                        final_result.append(output[j-1][i].cpu().numpy())
                        pg_final.append(pg_inputs[j-1][i].cpu().numpy())
                        label_final_result.append(label_inputs[j-1][i].cpu().numpy())
                        break
                    if(j==arities.size()[0]-1):
                        final_result.append(output[j][i].cpu().numpy())
                        pg_final.append(pg_inputs[j][i].cpu().numpy())
                        label_final_result.append(label_inputs[j][i].cpu().numpy())

    return np.exp(np.array(all_result).flatten()),np.exp(np.array(pg_all).flatten()),np.exp(np.array(label_all_result).flatten()),arities

def parse_plan(operators,plan_path):
    plan_trees = []
    with open(plan_path,'r') as f:
        lines = f.readlines()
    operator, in_operators = extract_operator(lines[0],operators)
    if not in_operators:
        operator, in_operators = extract_operator(lines[1],operators)
        j = 2
    else:
        j = 1
    if(operator == "Seq Scan"):
        pass
    else:
        while("actual" not in lines[j] and "Plan" not in lines[j]):
            j += 1
    current_node = Node(lines[j-1])
    plan_trees.append(current_node)
    spaces = 0
    node_stack = []
    i = j
    while not lines[i].startswith("Planning time"):
        line = lines[i]
        i += 1
        if line.startswith("Planning time") or line.startswith("Execution time"):
            break
        elif line.strip() == "":
            break
        elif ("->" not in line):
            continue
        else:
            if line.index("->") < spaces:
                while line.index("->") < spaces:
                    current_node, spaces = node_stack.pop()
            if line.index("->") > spaces:
                line_copy = line
                operator, in_operators = extract_operator(line_copy,operators)
                if(operator == "Seq Scan"):
                    pass
                else:
                    j = 0
                    while("actual" not in lines[i+j] and "Plan" not in lines[i+j]):
                        j += 1
                tokens = line_copy
                # print("token",tokens)
                new_node = Node(tokens, parent=current_node)
                current_node.add_child(new_node)
                node_stack.append((current_node, spaces))
                current_node = new_node
                spaces = line.index("->")
            elif line.index("->") == spaces:
                line_copy = line
                operator, in_operators = extract_operator(line_copy,operators)
                if(operator == "Seq Scan"):
                    pass
                else:
                    j = 0
                    while("actual" not in lines[i+j] and "Plan" not in lines[i+j]):
                        j += 1
                tokens = line_copy
                # print("token",tokens)
                new_node = Node(tokens, parent=node_stack[-1][0])
                node_stack[-1][0].add_child(new_node)
                current_node = new_node
                spaces = line.index("->")
    # print("scan count: ",scan_cnt)
    return plan_trees  # a list of the roots nodes

def plan_tree_to_relations(node):
    tree = {}
    tmp = node.data
    tree["line"] = tmp
    tree['children'] = []
    for children in node.children:
        tree['children'].append(plan_tree_to_relations(children))
    return tree

class OperatorEncoder(object):
    def __init__(self, value_fn, children_fn):
        super(OperatorEncoder, self).__init__()
        self.value_fn = value_fn
        self.children_fn = children_fn

    def encode(self, tree):
        children = self.children_fn(tree)
        n = len(children)
        value = self.value_fn(tree)

        if children:
            lower_values, lower_arities = \
                zip(* [self.encode(c) for c in reversed(children)])

            lower_values = list(lower_values)
            lower_arities = list(lower_arities)
        else:
            lower_values = []
            lower_arities = []
        lower_values.append(value)
        lower_arities.append(([len(children)]))
        return lower_values, lower_arities
        # lower_values.append(value.unsqueeze(0))
        # lower_arities.append(torch.LongTensor([len(children)]))
        # return torch.cat(lower_values), torch.cat(lower_arities)

def convert_tree_for_plan(father):
    t = (father['line'])
    # print(t)
    tmp = [t]
    tmp = tuple(tmp)
    # print(tmp)
    c = []
    for child in father['children']:
        c.append(tuple(convert_tree_for_plan(child)))
    tmp += (c,)
    return tmp

def extract_list(input):
    flatten = []
    for i in input:
        # print("extract: ",i)
        if(isinstance(i,list)):
            flatten.extend(extract_list(i))
        else:
            flatten.append(i)
    return flatten

def Q_error(pred,label):
    qerror = []
    for i in range(len(pred)):
        if pred[i]==0 and float(label[i])==0:
            qerror.append(1)
        elif pred[i]==0:
            qerror.append(label[i])
        elif label[i]==0:
            qerror.append(pred[i])
        elif pred[i] > float(label[i]):
            qerror.append(float(pred[i]) / float(label[i]))
        else:
            qerror.append(float(label[i]) / float(pred[i]))
    return np.array(qerror)

def get_relations_in_plan(plan):
    operators = ['Merge Join', 'Hash', 'Index Only Scan using title_pkey on title t', 'Sort','Seq Scan',\
        'Index Scan using title_pkey on title t', 'Materialize', 'Nested Loop', 'Hash Join']
    tree = parse_plan(operators,plan)
    # print(tree)
    tree_with_relations = plan_tree_to_relations(tree[0])
    # print("tree_with_relations: ",tree_with_relations)
    # ec = TreeEncoder(lambda x: x[0],lambda x: x[1])
    # converted_tree = convert_tree_for_plan(tree_with_relations)
    # inputs, arities = ec.encode_str([converted_tree])
    operator_ec = OperatorEncoder(lambda x:x['line'],lambda x: x['children'])
    inputs, arities = operator_ec.encode(tree_with_relations)
    inputs_flatten = extract_list(inputs)
    arities_flatten = extract_list(arities)
    # print("inputs flatten: ", inputs_flatten)
    # print("arities flatten: ",arities_flatten)
    return inputs_flatten,arities_flatten

def find_bad_estimation(predicted,pg,tree_node,arities,leaf=False,threshold=10):
    """find pg estimation with qerror large than threshold

    Args:
        predicted ([type]): [description]
        pg ([type]): [description]
        tree_node ([type]): [description]
        leaf (bool, optional): [description]. Defaults to False.
        threshold (int, optional): [description]. Defaults to 10.
    """
    # print(predicted)
    # print(pg)
    q_error = Q_error(predicted,pg)
    print((q_error>threshold))
    print(arities!=0)

    if(leaf==True):
        condition = q_error>threshold
    else:
        condition = (q_error>threshold) & (arities!=0)
    bad_estimation = []
    for i,c in enumerate(condition):
        if(c):
            bad_estimation.append(tree_node[i])
    # print("bad estimation: ",bad_estimation)
    return bad_estimation, condition

def construct_sql(table,card):
    return "set cardinality '{}' = {};\n".format(table,card)

def generate_sql(tree_node,predicted,bad_position,arities):
    operators = ['Merge Join', 'Seq Scan', 'Nested Loop', 'Hash Join']
    vocabulary = [ 'mk', 't', 'mi', 'mc', 'ci', 'mi_idx']

    tables = []
    set_cards = []
    ope_cnt = -1
    for pos in range(len(bad_position))[::-1]:
        if(bad_position[pos]==True):
            if(arities[pos] ==0):
                ope = tree_node[pos]
                table = ope.split("  (cost")[0].split(" ")[-1]
                tables.append(table)
                set_cards.append(predicted[pos])
                ope_cnt -= 1
            elif(arities[pos]==1):
                continue
            else:
                # find leaf node of the join
                children = 0
                COND = 2
                tmp_pos = pos
                joined_tables = []
                while(children<COND):
                    tmp_pos -= 1
                    if(arities[tmp_pos]==0):
                        children += 1
                        if("Seq Scan" in tree_node[tmp_pos]):
                            subtable = tree_node[tmp_pos].split("  (cost")[0].split(" ")[-1]
                        else:
                            subtable = "t"
                        joined_tables.append(subtable)
                    elif(arities[tmp_pos]==1):
                        continue
                    else:
                        COND += 1
                tables.append(",".join(joined_tables))
                set_cards.append(predicted[pos])
    print(tables)
    print(set_cards)
    queries = []
    for table,card in zip(tables,set_cards):
        queries.append(construct_sql(table,card))
    print(queries)
    return queries

def write_file(set_queries,raw_query,file_path):
    with open(file_path,"w") as f:
        f.writelines(set_queries)
        f.writelines(raw_query)
        f.writelines("\nreset cardinalities;")







if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Cardinality Error Detection & Injection')
    parser.add_argument("--plan",type=str,help="path of plan needed to be optimized",default="../data/example_plan.txt")
    parser.add_argument("--leaf-embedding-path",type=str,help="model path for leaf embedding",default="../model/embedding_model.h5")
    parser.add_argument("--TreeLSTM-model-path",type=str,help="model path for TreeLSTM",default="../model/treelstm_model")
    parser.add_argument("--save-path",type=str,help="file path for saving cardinality injecting queries",default="../data/injection_queries.txt")
    parser.add_argument("--query",type=str,help="file path or raw query starting with 'explain analyse...'",default="explain analyse select * from test;")
    args = parser.parse_args()

    plan = args.plan
    leaf_model_path = args.leaf_embedding_path
    treelstm_model_path = args.TreeLSTM_model_path
    file_path = args.save_path
    try:
        with open(args.query,"r") as f:
            raw_query = f.readlines()
    except:
        raw_query = args.query

    leaf_embedding = leaf_embedded(plan,model_path=leaf_model_path)
    # print("leaf embedding shape: ",np.shape(leaf_embedding))

    test_tree = tree_embedding(leaf_embedding,plan)

    predicted_all, pg_all, label_all, arities = predict(test_tree,treelstm_model_path)

    #map predictions with corresponding tree node
    tree_node, tree_arities = get_relations_in_plan(plan)

    # find bad_estimation
    bad_estimation, bad_position = find_bad_estimation(predicted_all,pg_all,tree_node,np.array(tree_arities))

    # generate set card queries
    set_queries = generate_sql(tree_node,predicted_all,bad_position,tree_arities)

    write_file(set_queries,raw_query,file_path)
    # TODO: check label and model