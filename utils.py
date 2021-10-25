import pandas as pd
import streamlit as st
import numpy as np
from google.cloud import firestore

import json
from google.cloud import firestore
from google.oauth2 import service_account


# def save_new_ratings(rating_dic, is_true_ratings, accuracy_check, user_id, pseudo, time, database = "data/new_database.csv"):
#     df = pd.read_csv(database)
#     for item in rating_dic.keys():
#         df = df.append({"userID":user_id,
#                         "pseudo": pseudo,
#                            "itemID":item,
#                            "rating":rating_dic[item],
#                            "is_true_ratings":is_true_ratings,
#                            "accuracy_check": accuracy_check,
#                          "time":time}, ignore_index=True)
#     df.to_csv(database, index=False)

def save_new_ratings(rating_dic, is_true_ratings, accuracy_check, user_id, pseudo, time, collection):
    for item in rating_dic.keys():
        doc_ref = collection.document()
        doc_ref.set({"userID":user_id,
        "pseudo": pseudo,
           "itemID":item,
           "rating":rating_dic[item],
           "is_true_ratings":is_true_ratings,
           "accuracy_check": accuracy_check,
         "time":time})



#@st.cache
def generate_user_id(dataset_path):
    #generate a user_id
    user_id = np.random.randint(1000, 1e8)
    df_clean = pd.read_csv(dataset_path)

    while user_id in df_clean["userID"]:
        user_id = np.random.randint(1000, 1e8)
    return user_id

@st.cache(hash_funcs={"_thread.RLock": lambda _:None, "builtins.weakref":lambda _:None, "google.cloud.firestore_v1.client.Client": lambda _:None})
def load_collection():
    key_dict = json.loads(st.secrets["textkey"])
    creds = service_account.Credentials.from_service_account_info(key_dict)
    db = firestore.Client(credentials=creds, project="nootropics-2a049")

    # Once the user has submitted, upload it to the database
    collection = db.collection("ratings")
    return collection
