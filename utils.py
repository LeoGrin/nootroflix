import pandas as pd
import streamlit as st
import numpy as np
from google.cloud import firestore
import json
from google.cloud import firestore
from google.oauth2 import service_account
import os


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

def save_new_ratings(rating_dic, issues_dic, question_dic, is_true_ratings, accuracy_check, user_id, pseudo, time, collection_ratings, collection_users):
    for item in rating_dic.keys():
        doc_ref = collection_ratings.document()
        doc_ref.set({"userID":user_id,
            "pseudo": pseudo,
           "itemID":item,
           "rating":rating_dic[item],
            "issue":issues_dic[item],
           "is_true_ratings":is_true_ratings,
           "accuracy_check": accuracy_check,
         "time":time})
        doc_ref_user = collection_users.document()
        user_dic = {"userID":user_id, "time":time, "pseudo":pseudo}
        user_dic.update(question_dic)
        doc_ref_user.set(user_dic)





@st.cache
def generate_user_id(dataset_path, session_id):
    #generate a user_id
    user_id = np.random.randint(1000, 1e8)
    df_clean = pd.read_csv(dataset_path)

    while user_id in df_clean["userID"]:
        user_id = np.random.randint(1000, 1e8)
    return user_id

@st.cache(ttl=600, hash_funcs={"_thread.RLock": lambda _:None, "builtins.weakref":lambda _:None, "google.cloud.firestore_v1.client.Client": lambda _:None})
def load_collection():
    keys = ["project_id", "type", "private_key_id", "private_key",
            "client_email", "client_id", "auth_uri", "token_uri",
            "auth_provider_x509_cert_url", "client_x509_cert_url"]
    cred_dic = {}
    for key in keys:
        cred_dic[key] = os.environ.get(key).replace("\\n", "\n")
    #key_dict = json.loads(st.secrets["textkey"])
    creds = service_account.Credentials.from_service_account_info(cred_dic)
    db = firestore.Client(credentials=creds, project="nootropics-2a049")

    # Once the user has submitted, upload it to the database
    collection_ratings = db.collection("ratings")
    collection_users = db.collection("users")
    return collection_ratings, collection_users
