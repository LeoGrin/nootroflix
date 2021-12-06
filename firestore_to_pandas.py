import pandas as pd
from google.cloud import firestore
from google.oauth2 import service_account
import os
from experiments.new_names import all_nootropics

# key_dict = json.loads(st.secrets["textkey"])
# for heroku
keys = ["project_id", "type", "private_key_id", "private_key",
        "client_email", "client_id", "auth_uri", "token_uri",
        "auth_provider_x509_cert_url", "client_x509_cert_url"]
cred_dic = {}
for key in keys:
    cred_dic[key] = os.environ.get(key).replace("\\n", "\n").replace("\\", "")
    print(cred_dic[key])

creds = service_account.Credentials.from_service_account_info(cred_dic)
db = firestore.Client(credentials=creds, project="nootropics-2a049")

users = list(db.collection(u'ratings').stream())

users_dict = list(map(lambda x: x.to_dict(), users))
df = pd.DataFrame(users_dict)

start_time = 1638206167
df = df[df["time"] > start_time]

print(len(df))
print(len(set(df["userID"].values)))

df = df[df["is_true_ratings"] == True]  # remove false ratings

to_join = df.groupby(['userID']).min("time")

to_join = to_join.rename({"time": "min_time"}, axis=1)[["min_time"]]

df = df.set_index("userID").join(to_join)

df = df[df["time"] == df["min_time"]]

df = df[["itemID", "rating"]].reset_index()

df_ssc = pd.read_csv("data/dataset_clean_right_names.csv")

total_df = df.append(df_ssc)

total_df = total_df[total_df["itemID"].isin(all_nootropics)]

# nootropics_with_enough_ratings = []
# for noot in all_nootropics:
#     if total_df[total_df["itemID"] == noot].shape[0] > 10: #TODO : optimize the number
#         nootropics_with_enough_ratings.append(noot)
#
# print(set(nootropics_with_enough_ratings).difference(set(all_nootropics)))
#
# total_df = total_df[total_df["itemID"].isin(nootropics_with_enough_ratings)]


total_df.to_csv("data/total_df.csv", index=False)

df = df[df["itemID"].isin(all_nootropics)]

df.to_csv("data/new_df.csv", index=False)  # only new ratings
