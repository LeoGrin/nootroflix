import pandas as pd
import numpy as np
from surprise import KNNBaseline
from surprise import Dataset
from surprise import Reader
from copy import deepcopy
import streamlit as st
from utils import load_database

def predict(rating_dic):
    df_clean = load_database()
    avalaible_nootropics = np.unique(df_clean["itemID"])  # we want to ignore nootropics that are not in the df
    #######################
    # Fit surprise model
    #######################

    # final_model = KNNBaseline(k=100, min_k=2, sim_options={'name': 'pearson_baseline', 'user_based': True})
    final_model = KNNBaseline(**{'verbose': False, 'k': 50, 'min_k': 5,  # check
                                 'sim_options': {'name': 'msd', 'user_based': False},
                                 'bsl_options': {'method': 'sgd', 'n_epochs': 500}})
    # final_model = SVD(**{'n_factors': 10, 'n_epochs': 20, 'lr_all': 0.005, 'reg_all': 0.1})

    new_user_id = max(df_clean["userID"]) + 1  # TODO if merge
    items = np.array([item for item in list(rating_dic.keys())])
    ratings = np.array([rating_dic[item] for item in items])
    rated_mask = ratings != None
    ratings = ratings[rated_mask]
    items = items[rated_mask]
    user = np.ones(len(items), dtype="int") * new_user_id
    new_user_df = pd.DataFrame({"userID": user, "itemID": items, "rating": ratings})

    total_df = df_clean.append(new_user_df)

    # A reader is still needed but only the rating_scale param is requiered.
    reader = Reader(rating_scale=(0, 10))

    # The columns must correspond to user id, item id and ratings (in that order).
    new_trainset = Dataset.load_from_df(total_df, reader).build_full_trainset()

    ## Fit the best model
    final_model.fit(new_trainset)

    predicted_ratings = []
    noot_to_rate = [noot for noot in avalaible_nootropics if noot not in rating_dic.keys()]
    for nootropic in noot_to_rate:
        predicted_ratings.append(final_model.predict(new_user_id, nootropic).est)
    # interpret_prediction(new_trainset, final_model, avalaible_nootropics, new_user_id, predicted_ratings, rating_dic, item_baselines_user)
    # mean_boost = np.median((predicted_ratings - item_baselines_user) / item_baselines_user)

    mean_rating = [np.mean(df_clean[df_clean["itemID"] == noot]["rating"]) for noot in noot_to_rate]
    result_df = pd.DataFrame(
        {"nootropic": noot_to_rate,
         "Prediction": predicted_ratings,
         "Mean rating": mean_rating})
    # "Boost": 100 * ((predicted_ratings - item_baselines_user) / item_baselines_user - mean_boost)})
    # "baseline_rating_user": item_baselines_user}) #TODO ?
    # mask = [noot not in rating_dic.keys() for noot in avalaible_nootropics]
    # result_df = result_df.iloc[mask]

    return result_df


def evaluate(rating_dic):
    df_clean = load_database()
    avalaible_nootropics = np.unique(df_clean["itemID"])  # we want to ignore nootropics that are not in the df
    loo_ratings = []
    rating_dic_copy = deepcopy(rating_dic)
    rated_avalaible_nootropics = [nootropic for nootropic in rating_dic.keys() if nootropic in avalaible_nootropics]
    # rated_avalaible_nootropics = np.random.choice(rated_avalaible_nootropics, min(len(rated_avalaible_nootropics), 10), replace=False)
    if len(rated_avalaible_nootropics) < 2:
        st.warning("Please rate more nootropics")
        return None
    else:
        for nootropic in rated_avalaible_nootropics:  # limit to 10 the number to prevent to much time to fit #TODO make it easier to compute
            rating_dic_copy.pop(nootropic)
            new_result_df = predict(rating_dic_copy)
            loo_ratings.append(new_result_df[new_result_df["nootropic"] == nootropic]["Prediction"].values[0])
            rating_dic_copy = deepcopy(rating_dic)

    # item_baselines_df = get_item_baseline()
    # item_baselines = item_baselines_df[item_baselines_df["nootropic"].isin(rating_dic.keys())]["item_baselines"].values
    mean_rating = [np.mean(df_clean[df_clean["itemID"] == noot]["rating"]) for noot in rated_avalaible_nootropics]

    return pd.DataFrame({"nootropic": [noot for noot in rated_avalaible_nootropics],
                         "Your rating": [rating_dic[nootropic] for nootropic in rated_avalaible_nootropics],
                         "Predicted rating": loo_ratings,
                         "Mean rating": mean_rating})