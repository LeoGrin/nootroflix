import pandas as pd

df = pd.DataFrame({"userID": [],
                   "itemID": [],
                   "rating": [],
                   "is_true_ratings": [],
                   "accuracy_check": []})
df.to_csv("new_database.csv")