import pandas as pd
from surprise import Reader, Dataset
import numpy as np
import matplotlib.pyplot as plt

OVERWRITE = False

## Algorithm selection

# %%

df_clean = pd.read_csv("../data/total_df.csv")

#plt.hist(df_clean["rating"], bins=11)
#plt.show()

print(np.unique(df_clean["rating"], return_counts=True))


min_max = df_clean.groupby("userID").agg([np.min, np.max], "rating")["rating"]

plt.hist(min_max["amin"])
plt.show()

plt.hist(min_max["amax"])
plt.show()

# A reader is still needed but only the rating_scale param is requiered.
reader = Reader(rating_scale=(0, 10))

# The columns must correspond to user id, item id and ratings (in that order).
data = Dataset.load_from_df(df_clean, reader)

# %%
