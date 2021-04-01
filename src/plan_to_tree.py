import os
import numpy as np


scan_features_path = '../data/...npy'


operators = ['Merge Join', 'Hash', 'Index Only Scan using title_pkey on title t', 'Sort','Seq Scan', 'Index Scan using title_pkey on title t', 'Materialize', 'Nested Loop', 'Hash Join']
columns = ['ci.movie_id', 't.id', 'mi_idx.movie_id', 'mi.movie_id', 'mc.movie_id', 'mk.movie_id']
scan_features = np.load("./final/job-light_scan_features_64.npy")
print(len(operators))