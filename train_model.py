import pandas as pd
import numpy as np
from surprise import KNNBaseline
from surprise import Dataset
from surprise import Reader
from copy import deepcopy
import streamlit as st
from utils import load_database

@st.experimental_singleton
def compute_mean_ratings():
    df_clean = load_database()
    df_clean = df_clean.groupby(["itemID"])['rating'].mean()
    return df_clean.to_dict()


@st.experimental_singleton
def train_model():
    # train the model once and save everything we need in cache
    df_clean = load_database()
    avalaible_nootropics = np.unique(df_clean["itemID"])
    final_model = KNNBaseline(**{'verbose': False, 'k': 50, 'min_k': 5,  # check
                                 'sim_options': {'name': 'msd', 'user_based': False},
                                 'bsl_options': {'method': 'sgd', 'n_epochs': 500}})
    reader = Reader(rating_scale=(0, 10))
    new_trainset = Dataset.load_from_df(df_clean, reader).build_full_trainset()
    final_model.fit(new_trainset)
    item_baselines_inner = final_model.default_prediction() + final_model.compute_baselines()[
        1]  # mean rating + item baseline ?
    similarity_matrix = final_model.compute_similarities()
    return avalaible_nootropics, item_baselines_inner, similarity_matrix, new_trainset.to_inner_iid


# def interpret_prediction(trainset, model, avalaible_nootropics, user_id, predicted_ratings, rating_dic, item_baselines_user):
#     # Disappointing results for now, try again with more data
#     similarity_matrix = model.compute_similarities()
#     print("ritalin adderall")
#     print(similarity_matrix[trainset.to_inner_iid("Methylphenidate (Ritalin)"), trainset.to_inner_iid("Adderall")])
#     #similarity_matrix -= np.eye(len(similarity_matrix))
#     indices_available_nootropics = np.array([trainset.to_inner_iid(item) for item in avalaible_nootropics])
#     similarity_matrix = similarity_matrix[indices_available_nootropics][:, indices_available_nootropics]
#
#     for i, nootropic in enumerate(avalaible_nootropics):
#         print("########")
#         print("Nootropic: " + nootropic)
#         predicted_ratings.append(model.predict(user_id, nootropic).est)
#         similarities = similarity_matrix[np.where(indices_available_nootropics == trainset.to_inner_iid(nootropic))[0]].reshape(-1)
#         #similarities = similarity_matrix[trainset.to_inner_iid(nootropic)].reshape(-1)
#         importances = np.zeros(len(similarities))
#         sim_sum = 0
#         for item in rating_dic:
#             if not rating_dic[item] is None and item != nootropic and item in avalaible_nootropics:
#                 indice_item = np.where(indices_available_nootropics == trainset.to_inner_iid(item))[0]
#                 #indice_item = trainset.to_inner_iid(item)
#                 #print(item)
#                 #print(rating_dic[item])
#                 #print(item_baselines_user[indice_item])
#                 sim_sum += similarities[indice_item]
#                 #print(np.abs(similarities[indice_item] * (rating_dic[item] - item_baselines_user[indice_item])))
#                 importances[indice_item] = similarities[indice_item]# * (rating_dic[item] - item_baselines_user[indice_item])
#         importances = importances / sim_sum
#         ind = np.argsort(np.abs(importances))[::-1][:20]
#         print([trainset.to_raw_iid(i) for i in ind])
#         print(importances[ind])
#         print(item_baselines_user[np.where(indices_available_nootropics == trainset.to_inner_iid(nootropic))[0]])
#         print(item_baselines_user[np.where(indices_available_nootropics == trainset.to_inner_iid(nootropic))[0]] + np.sum(importances))
#         print(predicted_ratings[i])

def predict(rating_dic):
    """
    Predict the ratings of the nootropics for the user
    We compute it by hand instead of using the Surprise model.predict function to avoid refitting the model for each user
    :param rating_dic: a dictionary of the form {itemID: rating}
    :return: DataFrame containing itemID, predictions, mean_ratings
    """
    avalaible_nootropics, item_baselines_inner, similarity_matrix, raw_to_iid = train_model()
    mean_ratings_dic = compute_mean_ratings()

    user_baseline = np.mean([rating_dic[a] - item_baselines_inner[raw_to_iid(a)] for a in rating_dic.keys()])
    user_baseline /= (1 + 0.02)
    # print(final_model.compute_baselines()[0][-1])

    predicted_ratings = []
    noot_to_rate = [noot for noot in avalaible_nootropics if noot not in rating_dic.keys()]
    for nootropic in noot_to_rate:
        inner_id = raw_to_iid(nootropic)
        pred = user_baseline + item_baselines_inner[inner_id]
        to_add = 0
        n_neighbors_used = 0
        sim_sum = 0
        similarities = [similarity_matrix[inner_id, raw_to_iid(item)] for item in rating_dic.keys()]
        for idx in np.argsort(similarities)[::-1][:50]:
            item = list(rating_dic.keys())[idx]
            id_item = raw_to_iid(item)
            if similarities[idx] > 0:
                to_add += similarities[idx] * (rating_dic[item] - item_baselines_inner[id_item] - user_baseline)
                n_neighbors_used += 1
                sim_sum += similarities[idx]
        if n_neighbors_used >= 5:
            pred += to_add / sim_sum
        if pred < 0:
            pred = 0
        if pred > 10:
            pred = 10
        predicted_ratings.append(pred)

    mean_ratings = [mean_ratings_dic[a] for a in noot_to_rate]

    return pd.DataFrame({"nootropic": noot_to_rate,
                         "Prediction": predicted_ratings,
                         "Mean rating": mean_ratings})

def evaluate(rating_dic):
    avalaible_nootropics, item_baselines_inner, similarity_matrix, raw_to_iid = train_model()
    mean_ratings_dic = compute_mean_ratings()
    rated_avalaible_nootropics = [nootropic for nootropic in rating_dic.keys() if nootropic in avalaible_nootropics]
    loo_ratings = []
    # Predict without refitting, and without considering one rating each time (for the user baseline and for the similarities)
    for nootropic_to_remove in rated_avalaible_nootropics:
        user_baseline = np.mean([rating_dic[a] - item_baselines_inner[raw_to_iid(a)] for a in rating_dic.keys() if
                                 a != nootropic_to_remove])
        user_baseline /= (1 + 0.02)
        inner_id = raw_to_iid(nootropic_to_remove)
        pred = user_baseline + item_baselines_inner[inner_id]
        to_add = 0
        n_neighbors_used = 0
        sim_sum = 0
        similarities = [similarity_matrix[inner_id, raw_to_iid(item)] for item in rating_dic.keys()]
        for idx in np.argsort(similarities)[::-1][:50]:
            item = list(rating_dic.keys())[idx]
            id_item = raw_to_iid(item)
            if item != nootropic_to_remove:
                if similarities[idx] > 0:
                    to_add += similarities[idx] * (rating_dic[item] - item_baselines_inner[id_item] - user_baseline)
                    n_neighbors_used += 1
                    sim_sum += similarities[idx]
        if n_neighbors_used >= 5:
            pred += to_add / sim_sum
        if pred < 0:
            pred = 0
        if pred > 10:
            pred = 10
        loo_ratings.append(pred)
    mean_ratings = [mean_ratings_dic[noot] for noot in rated_avalaible_nootropics]
    return pd.DataFrame({"nootropic": [noot for noot in rated_avalaible_nootropics],
                         "Your rating": [rating_dic[nootropic] for nootropic in rated_avalaible_nootropics],
                         "Predicted rating": loo_ratings,
                         "Mean rating": mean_ratings})


if __name__ == """__main__""":
    rating_example = {'Modafinil': 6,
                      'Caffeine': 6,
                      'Coluracetam': None,
                      'Phenylpiracetam': None,
                      'Theanine': 7,
                      'Noopept': None,
                      'Oxiracetam': None,
                      'Aniracetam': None,
                      'Rhodiola': None,
                      'Creatine': 4,
                      'Piracetam': None,
                      'Ashwagandha': None,
                      'Bacopa': None,
                      'Choline': None,
                      'DMAE': None,
                      'Fasoracetam': None,
                      'SemaxandNASemaxetc': None,
                      'SelankandNASelanketc': None,
                      'Inositol': None,
                      'Seligiline': None,
                      'AlphaBrainproprietaryblend': None,
                      'Cerebrolysin': None,
                      'Melatonin': 0,
                      'Uridine': None,
                      'Tianeptine': None,
                      'MethyleneBlue': None,
                      'Unifiram': None,
                      'PRL853': None,
                      'Emoxypine': None,
                      'Picamilon': None,
                      'Dihexa': None,
                      'Epicorasimmunebooster': None,
                      'Epicorasimmunebooster': None,
                      'LSD': 0,
                      'Adderall': 8,
                      "Phenibut": 6,
                      "Nicotine": 7}
    print(predict(rating_example))
