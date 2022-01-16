import pandas as pd
from google.cloud import firestore
from google.oauth2 import service_account
import os


LOAD_USERS = True
LOAD_RATINGS = True
LOAD_POSITIONS = False
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

if LOAD_USERS:
    users = list(db.collection(u'users').stream())
    users_dict = list(map(lambda x: x.to_dict(), users))
    df_users = pd.DataFrame(users_dict)
    print(df_users)
    #df_users.to_csv("data/users.csv")

if LOAD_RATINGS:

    ratings = list(db.collection(u'ratings').stream())
    ratings_dict = list(map(lambda x: x.to_dict(), ratings))
    df = pd.DataFrame(ratings_dict)



    start_time = 1638206167
    df = df[df["time"] > start_time]

    print('Without SSC')
    print(len(df))
    print(len(set(df["userID"].values)))

    df = df[df["is_true_ratings"] == True]  # remove false ratings

    print("Without ssc, true ratings")
    print(len(df))
    print(len(set(df["userID"].values)))

    to_join = df.groupby(['userID']).min("time")

    to_join = to_join.rename({"time": "min_time"}, axis=1)[["min_time"]]

    df = df.set_index("userID").join(to_join)

    df = df[df["time"] == df["min_time"]]

    df = df[["itemID", "rating", "issue", "time"]].reset_index()

    df_ssc = pd.read_csv("data/dataset_clean_right_names.csv")

    total_df = df.append(df_ssc)

    print("With SSC, true ratings")
    print(len(total_df))
    print(len(set(total_df["userID"].values)))


    #convert old names to new names
    translation_dic = {}
    for i, row in pd.read_csv("data/nootropics_metadata.csv", sep=";").iterrows():
        for old_name in row["old_names"].split(";"):
            translation_dic[old_name] = row["nootropic"]

    total_df["itemID"] = list(map(lambda x: translation_dic[x], total_df["itemID"]))

    #total_df = total_df[total_df["itemID"].isin(all_nootropics)]

    # nootropics_with_enough_ratings = []
    # for noot in all_nootropics:
    #     if total_df[total_df["itemID"] == noot].shape[0] > 10: #TODO : optimize the number
    #         nootropics_with_enough_ratings.append(noot)
    #
    # print(set(nootropics_with_enough_ratings).difference(set(all_nootropics)))
    #
    # total_df = total_df[total_df["itemID"].isin(nootropics_with_enough_ratings)]


    total_df[["userID", "itemID", "rating"]].to_csv("data/total_df.csv", index=False)


    df["itemID"] = list(map(lambda x: translation_dic[x], df["itemID"]))

    df[["userID", "itemID", "rating"]].to_csv("data/new_df.csv", index=False)  # only new ratings

    df.to_csv("data/new_df_full.csv", index=False)  # only new ratings

if LOAD_POSITIONS:
    positions = list(db.collection(u'position').stream())
    positions_dict = list(map(lambda x: x.to_dict(), positions))
    df_positions = pd.DataFrame(positions_dict)
    df_positions.to_csv("data/positions.csv")