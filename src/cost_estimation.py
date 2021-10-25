import numpy as np
import argparse
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import csv
import torch
import os
from torch import nn
from sklearn import preprocessing
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from blitz.modules import BayesianLSTM
from blitz.utils import variational_estimator

from sklearn.preprocessing import StandardScaler
from joblib import dump, load

def extract_plan(line):
    data = line.replace("->","").lstrip().split("  ")[-1].split(" ")
    start_cost = data[0].split("..")[0].replace("(cost=","")
    end_cost = data[0].split("..")[1]
    rows = data[1].replace("rows=","")
    width = data[2].replace("width=","").replace(")","")
    # a_start_cost = data[4].split("..")[0].replace("time=","")
    # a_end_cost = data[4].split("..")[1]
    # a_rows = data[5].replace("rows=","") 
    return float(start_cost),float(end_cost),float(rows),float(width)#,float(a_start_cost),float(a_end_cost),float(a_rows)

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
    for i in range(len(plan)):
        if("Seq Scan" in plan[i]):
            _start_cost,_end_cost,_rows,_width = extract_plan(plan[i])
            if(len(plan[i].strip().split("  "))==2):
                _sentence = " ".join(plan[i].strip().split("  ")[0].split(" ")[:-1]) + " "
                table = plan[i].strip().split("  ")[0].split(" ")[4]
            else:
                _sentence = " ".join(plan[i].strip().split("  ")[1].split(" ")[:-1]) + " "
                table = plan[i].strip().split("  ")[1].split(" ")[4]
            if("actual" not in plan[i+1] and "Plan" not in plan[i+1] and "->" not in plan[i+1]):
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
            rows.append(0)
            pg.append(_rows)
    # print(sentences)
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

def padding_sentence(raw_sentences,test_data,max_len=9):
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
                                    outputs=model.layers[4].output)
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
    feature_len = 9+6+4+64
    with open(plan_path,'r') as f:
        lines = f.readlines()
    feature_vec = [0.0]*feature_len
    operator, in_operators = extract_operator(lines[0],operators)
    # print("operator: ",operator)
    if not in_operators:
        operator, in_operators = extract_operator(lines[1],operators)
        start_cost, end_cost, rows, width = extract_plan(lines[1])
        j = 2
    else:
        start_cost, end_cost, rows, width = extract_plan(lines[0])
        j = 1
    feature_vec[feature_len-7:feature_len] = [start_cost, end_cost, rows, width]
    feature_vec[operators.index(operator)] = 1.0
    if(operator == "Seq Scan"):
        extract_attributes(operator, operators,columns,lines[j], feature_vec,leaf_embedding, scan_cnt)
        scan_cnt += 1
        # root_tokens = feature_vec
        # current_node = Node(root_tokens)
        # plan_trees.append(current_node)
    else:
        while("->" not in lines[j] and j<len(lines)):
            extract_attributes(operator, operators,columns, lines[j], feature_vec,leaf_embedding)
            j += 1
    root_tokens = feature_vec  # 所有吗
    current_node = Node(root_tokens)
    plan_trees.append(current_node)
    spaces = 0
    node_stack = []
    i = j
    while not i>=len(lines):
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
                start_cost, end_cost, rows, width = extract_plan(
                    line_copy)
                feature_vec[feature_len-7:feature_len] = [start_cost, end_cost, rows, width]
                operator, in_operators = extract_operator(line_copy,operators)
                feature_vec[operators.index(operator)] = 1.0
                if(operator == "Seq Scan"):
                    extract_attributes(
                        operator,  operators, columns, line_copy, feature_vec,leaf_embedding, scan_cnt)
                    scan_cnt += 1
                else:
                    j = 0
                    while("->" not in lines[i+j] and (i+j)<len(lines)):
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
                start_cost, end_cost, rows, width = extract_plan(
                    line_copy)
                feature_vec[feature_len-7:feature_len] = [start_cost, end_cost, rows, width]
                operator, in_operators = extract_operator(line_copy,operators)
                feature_vec[operators.index(operator)] = 1.0
                if(operator == "Seq Scan"):
                    extract_attributes(
                        operator, operators, columns, line_copy, feature_vec,leaf_embedding, scan_cnt)
                    scan_cnt += 1
                else:
                    j = 0
                    while("->" not in lines[i+j] and (i+j)<len(lines)):
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

def plan2seq(node):
    sequence = []
    tmp = node.data
    operators_count = 9
    columns_count = 6
    scan_features = 64
    if(len(node.children)!=0):
        for i in range(len(node.children)):
            sequence.extend(plan2seq(node.children[i]))
    sequence.append(tmp[:operators_count + columns_count+scan_features])
    return sequence

def tree_embedding(leaf_embedding,plan):
    operators = ['Merge Join', 'Hash', 'Index Only Scan using title_pkey on title t', 'Sort','Seq Scan',\
            'Index Scan using title_pkey on title t', 'Materialize', 'Nested Loop', 'Hash Join']
    columns = ['ci.movie_id', 't.id', 'mi_idx.movie_id', 'mi.movie_id', 'mc.movie_id', 'mk.movie_id']

    root_node, max_children = parse_tree(operators,columns,leaf_embedding,plan)
    # print(root_node)
    embedded_tree = plan2seq(root_node[0])
    return embedded_tree

@variational_estimator
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.lstm_1 = BayesianLSTM(79, 10, prior_sigma_1=1, prior_pi=1, posterior_rho_init=-3.0)
        self.linear = nn.Linear(10, 1)
            
    def forward(self, x):
        x_, _ = self.lstm_1(x)
        
        #gathering only the latent end-of-sequence for the linear layer
        x_ = x_[:, -1, :]
        x_ = self.linear(x_)
        return x_

def pred_cost(X,sample_nbr=100):
    global sc
    preds = [net(X).cpu().item() for i in range(sample_nbr)]
    pred = np.mean(preds)
    print(pred)
    pred = sc.inverse_transform(pred.reshape(1,1))[0][0]
    preds = sc.inverse_transform(preds)
    return pred,preds

def evaluate_preds(preds, scaled_y, std_multiplier=2):
    global sc
    # print(scaled_y)
    y = sc.inverse_transform(scaled_y.reshape(1,1))[0][0]
    mean = np.mean(preds)
    std = np.std(preds)
    ci_upper = mean + (std_multiplier * std)
    ci_lower = mean - (std_multiplier * std)
    ic_acc = (ci_lower <= y) * (ci_upper >= y)
    return y, ic_acc, (ci_upper >= y), (ci_lower <= y)

def get_intervals(preds,std_multiplier=2):
    
    mean = np.mean(preds)
    std = np.std(preds)
    
    upper_bound = mean + (std * std_multiplier)
    lower_bound = mean - (std * std_multiplier)

    
    return upper_bound,lower_bound

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Cardinality Error Detection & Injection')
    parser.add_argument("--plan-dir",type=str,help="path of plan needed to be optimized",default="../data/plan/9")
    parser.add_argument("--leaf-embedding-path",type=str,help="model path for leaf embedding",default="../model/embedding_model.h5")
    parser.add_argument("--cost-model-path",type=str,help="model path for TreeLSTM",default="../model/cost_model")
    parser.add_argument("--save-path",type=str,help="file path for saving cardinality injecting queries",default="../data/injection_queries.txt")
    args = parser.parse_args()

    plan_dir = args.plan_dir
    leaf_model_path = args.leaf_embedding_path
    cost_model_path = args.cost_model_path
    file_path = args.save_path

    sc = joblib.load("../model/std_scaler.bin")
    net = torch.load(cost_model_path)
    max_length = 8

    candidate_plans_num = len(os.listdir(plan_dir))
    for idx in range(candidate_plans_num):
        print(idx)
        cur_plan_path = os.path.join(plan_dir,str(idx))
        leaf_embedding = leaf_embedded(cur_plan_path,model_path=leaf_model_path)
        # print("leaf embedding shape: ",np.shape(leaf_embedding))
        test_tree = tree_embedding(leaf_embedding,cur_plan_path)

        if(len(test_tree))<max_length:
            tmp = [[0] * 79] * (max_length-len(test_tree))
            tmp.extend(test_tree)
            padded_sequences = tmp
        else:
            padded_sequences = test_tree
        padded_sequences = np.array(padded_sequences)
        padded_sequences = torch.tensor(padded_sequences,dtype=torch.float32)

        pred,preds = pred_cost(padded_sequences.unsqueeze(0))

        upper, lower = get_intervals(preds)

        # print("label: ", unscaled_y)
        print("prediction: ",pred)
        print("prediciton upper bound: ", upper)
        print("prediction lower bound: ", lower)
        print("*"*30)
        # print("label in prediction range: ",in_range)

        break
