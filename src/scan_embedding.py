import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model, load_model, Model
from tensorflow.keras import optimizers, activations
from tensorflow.keras.layers import Dense, Flatten, LSTM, Bidirectional, SimpleRNN,Dropout
from tensorflow.keras.losses import logcosh
# from tensorflow.metrics import mean_relative_error
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import LeakyReLU
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import os
import csv
from tensorflow.keras import regularizers
from matplotlib.pyplot import MultipleLocator
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)




file_name_column_min_max_vals = "../data/column_min_max_vals.csv"
# plan_path = "../data/JOB/cardinality/"
# test_path = "../data/JOB/synthetic/"
# test_path = "../data/JOB/job-light/'
# test_path = "../data/JOB/cardinality/"
plan_path = "/data/sunluming/datasets/JOB/cardinality"
test_path = plan_path

def extract_time(line):
    data = line.replace("->","").lstrip().split("  ")[-1].split(" ")
    start_cost = data[0].split("..")[0].replace("(cost=","")
    end_cost = data[0].split("..")[1]
    rows = data[1].replace("rows=","")
    width = data[2].replace("width=","").replace(")","")
    a_start_cost = data[4].split("..")[0].replace("time=","")
    a_end_cost = data[4].split("..")[1]
    a_rows = data[5].replace("rows=","") 
    return float(start_cost),float(end_cost),float(rows),float(width),float(a_start_cost),float(a_end_cost),float(a_rows)

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

def get_data_and_label(path):
    plans = sorted(os.listdir(path))
    sentences = []
    rows = []
    pg = []
    d = {}
    for file in sorted(plans):
        with open(path+'/'+file,'r') as f:
            plan = f.readlines()
            for i in range(len(plan)-2):
                if("Seq Scan" in plan[i]):
                    _start_cost,_end_cost,_rows,_width,_a_start_cost_,_a_end_cost,_a_rows = extract_time(plan[i])
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

def prepare_data_and_label(sentences,rows):
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
                # print(word)
                _tmp = np.full((vocab_size+1),word)
                # _tmp = np.column_stack((np.array([float(word)]),np.zeros((1,vocab_size))))
                # _tmp = np.reshape(_tmp,(vocab_size+1))
                # print(_tmp)
                assert(len(_tmp)==vocab_size+1)
                _s.append(_tmp)
        data.append(np.array(_s))
        label.append(row)
    return data,label

def normalize_labels(labels, min_val=None, max_val=None):
    # log tranformation withour normalize
    labels = np.array([np.log(float(l)) for l in labels]).astype(np.float32)
    return labels,0,1



with open(file_name_column_min_max_vals, 'r') as f:
    data_raw = list(list(rec) for rec in csv.reader(f, delimiter=','))
    column_min_max_vals = {}
    for i, row in enumerate(data_raw[1:]):
        column_min_max_vals[row[0]] = [float(row[1]), float(row[2])]


sentences,rows,pg = get_data_and_label(plan_path)

vocabulary = []
for sentence in sentences:
    for word in sentence:
        if(word not in vocabulary and is_not_number(word)):
            vocabulary.append(word)
# print(len(vocabulary))
vocab_size = len(vocabulary)
# print(vocabulary)

_vocabulary = np.array(vocabulary)
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(_vocabulary)
encoded = to_categorical(integer_encoded)
vocab_dict = {}
for v,e in zip(vocabulary,encoded):
    vocab_dict[v] = np.reshape(np.array(e),(1,vocab_size))

data,label = prepare_data_and_label(sentences,rows)
label_norm, min_val, max_val = normalize_labels(label)

max_len = 0
for sentence in sentences:
    if(len(sentence) > max_len):
        max_len = len(sentence)
print(max_len)
padded_sentences = pad_sequences(data, maxlen=max_len, padding='post',dtype='float32')

# print(np.shape(padded_sentences))
# print(np.shape(label_norm))

X_train, X_test, y_train, y_test = train_test_split(padded_sentences, label_norm, test_size=0.8, random_state=40)
pg_train,pg_test,_y_train,_y_test = train_test_split(pg,label_norm,test_size=0.2,random_state=40)
print(np.shape(X_train),np.shape(X_test))
print(np.shape(y_train),np.shape(y_test))


model = Sequential()
model.add(SimpleRNN(128, return_sequences=True,activation='relu',input_shape=(max_len,vocab_size+1)))
model.add(SimpleRNN(128,return_sequences=True,activation='relu'))
model.add(Flatten())
model.add(Dense(128,activation='relu'))#,kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(64,activation='relu'))#,kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(1, activation='relu'))

model.compile(optimizer=optimizers.Adagrad(lr=0.001), loss='mse', metrics=['mse','mae'])

model.summary()

model.fit(padded_sentences,label_norm,validation_split=0.2,epochs=100, batch_size=128,shuffle=True)

model.save('../model/embedding_model.h5') 
# model.load_weights("../model/embedding_model.h5")

# For validation & feature extraction
test_sentences,test_rows,test_pg = get_data_and_label(test_path)
test_data,test_label = prepare_data_and_label(test_sentences,test_rows)
print(np.shape(test_data),np.shape(test_label))


test_padded_sentences = pad_sequences(test_data, maxlen=max_len, padding='post',dtype='float32')

intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.layers[4].output)
intermediate_output = intermediate_layer_model.predict(test_padded_sentences)

np.save("../data/{}.npy".format(test_path.split("/")[-2]),intermediate_output)


# For vector demo

scan_label = []
scan_label.append(['table','detail'])
for sentence in sentences:
    tmp = []
#     print(sentence)
    table = sentence[0]
    tmp.append(table)
    if(len(sentence)>2):
        tmp.append(' '.join(str(each) for each in sentence[2:]))
    else:
        tmp.append(table)
    scan_label.append(tmp)
#     break

np.savetxt("../data/vectors.csv", intermediate_output, delimiter="\t")
np.savetxt("../data/labels.csv",scan_label,fmt='%s',delimiter = "\t")
