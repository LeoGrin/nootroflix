import pandas as pd
import numpy as np

df_clean = pd.read_csv("data/dataset_clean_right_names.csv")
avalaible_nootropics = np.unique(df_clean["itemID"])

print(avalaible_nootropics)