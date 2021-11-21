import pandas as pd
import numpy as np

df = pd.read_csv("../data/dataset_clean.csv")

print(df.pivot(columns = "itemID", values="rating").values)