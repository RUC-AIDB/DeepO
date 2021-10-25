# %%
import joblib
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from blitz.modules import BayesianLSTM
from blitz.utils import variational_estimator

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
# %%
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
# %%
with open("../data/job-cardinality-sequence.pkl","rb") as f:
    sequences = pickle.load(f)
cost_label = np.load("../data/cost_label.npy").reshape(-1,1)

sc = joblib.load("../model/std_scaler.bin")

cost_label = sc.fit_transform(cost_label)

net = torch.load("../model/cost_model")

max_length = 0
for sequence in sequences:
    if(np.shape(sequence)[0]>max_length):
        max_length = np.shape(sequence)[0]
# %%
padded_sequences = []
for seq in sequences:
    if(len(seq))<max_length:
        tmp = [[0] * 79] * (max_length-len(seq))
        tmp.extend(seq)
        padded_sequences.append(tmp)
    else:
        padded_sequences.append(seq)
padded_sequences = np.array(padded_sequences)
# %%
padded_sequences = torch.tensor(padded_sequences,dtype=torch.float32)
cost_label = torch.tensor(cost_label,dtype=torch.float32)


# %%
X_train, X_test, y_train, y_test = train_test_split(padded_sequences,
                                                    cost_label,
                                                    test_size=.25,
                                                    random_state=42,
                                                    shuffle=False)

# %%
def pred_cost(X,sample_nbr=100):
    global sc
    preds = [net(X).cpu().item() for i in range(sample_nbr)]
    pred = np.mean(preds)
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
# %%

idx = 2
pred,preds = pred_cost(X_test[idx].unsqueeze(0))

unscaled_y, in_range, y_under_upper, y_above_lower = evaluate_preds(preds,y_test[idx],2)

upper, lower = get_intervals(preds)

print("label: ", unscaled_y)
print("prediction: ",pred)
print("prediciton upper bound: ", upper)
print("prediction lower bound: ", lower)
print("label in prediction range: ",in_range)
# %%
cnt = 0
for idx in range(len(X_test)):
    pred,preds = pred_cost(X_test[idx].unsqueeze(0))
    unscaled_y, in_range, y_under_upper, y_above_lower = evaluate_preds(preds,y_test[idx],2)
    if(in_range):
        cnt += 1
print(cnt/len(X_test))

# %%
# 63608% in range