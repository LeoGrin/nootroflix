import lightfm
import lightfm.data
import lightfm.evaluation
from lightfm import LightFM
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix


df = pd.read_csv('../data/total_df.csv')
print(df)

dataset = lightfm.data.Dataset()
dataset.fit(np.unique(df["userID"].values), np.unique(df["itemID"].values))

num_users, num_items = dataset.interactions_shape()
print('Num users: {}, num_items {}.'.format(num_users, num_items))

tuple_list = []
for i, row in df.iterrows():
    tuple_list.append(row.values)

#print(tuple_list)

#%%

(interactions, weights) = dataset.build_interactions(tuple_list)

#%%

model = LightFM()
model.fit(weights)

#%%

print(lightfm.evaluation.auc_score(model, weights))
print(np.mean(lightfm.evaluation.auc_score(model, weights)))


#%%
