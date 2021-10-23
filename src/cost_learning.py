# %%
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

# %%

with open("../data/job-cardinality-sequence.pkl","rb") as f:
    sequences = pickle.load(f)
cost_label = np.load("../data/cost_label.npy").reshape(-1,1)
# %%
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

ds = torch.utils.data.TensorDataset(X_train, y_train)
dataloader_train = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True)    

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
net = NN()
net = net.float()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
# %%
iteration = 0
for epoch in range(10):
    for i, (datapoints, labels) in enumerate(dataloader_train):
        optimizer.zero_grad()
        
        loss = net.sample_elbo(inputs=datapoints,
                               labels=labels,
                               criterion=criterion,
                               sample_nbr=3,
                               complexity_cost_weight=1/X_train.shape[0])
        loss.backward()
        optimizer.step()
        
        iteration += 1
        if iteration%250==0:
            preds_test = net(X_test)[:,0].unsqueeze(1)
            loss_test = criterion(preds_test, y_test)
            print("Iteration: {} Val-loss: {:.4f}".format(str(iteration), loss_test))
# %%
