import pandas as pd
import streamlit as st
import numpy as np


def save_new_ratings(rating_dic, is_true_ratings, accuracy_check, user_id, pseudo, time, database = "data/new_database.csv"):
    df = pd.read_csv(database)
    for item in rating_dic.keys():
        df = df.append({"userID":user_id,
                        "pseudo": pseudo,
                           "itemID":item,
                           "rating":rating_dic[item],
                           "is_true_ratings":is_true_ratings,
                           "accuracy_check": accuracy_check,
                         "time":time}, ignore_index=True)
    df.to_csv(database, index=False)


@st.cache
def generate_user_id(dataset_path):
    #generate a user_id
    user_id = np.random.randint(1000, 1e8)
    df_clean = pd.read_csv(dataset_path)

    while user_id in df_clean["userID"]:
        user_id = np.random.randint(1000, 1e8)
    return user_id