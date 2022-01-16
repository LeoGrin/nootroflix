import pandas as pd
import numpy as np
from surprise import KNNBaseline
from surprise import Dataset
from surprise import Reader
from copy import deepcopy
import streamlit as st
from utils import load_database




# def get_item_baseline():
#     df_clean = pd.read_csv("data/dataset_clean_right_names.csv")
#     avalaible_nootropics = np.unique(df_clean["itemID"])
#
#     final_model = KNNBaseline(k=100, min_k=2, sim_options={'name': 'pearson_baseline', 'user_based': True})
#     #final_model = SVD(**{'n_factors': 10, 'n_epochs': 20, 'lr_all': 0.005, 'reg_all': 0.1})
#
#
#     total_df = df_clean
#
#     # A reader is still needed but only the rating_scale param is requiered.
#     reader = Reader(rating_scale=(0, 10))
#
#     # The columns must correspond to user id, item id and ratings (in that order).
#     new_trainset = Dataset.load_from_df(total_df, reader).build_full_trainset()
#
#     ## Fit the best model
#
#     final_model.fit(new_trainset)
#
#     item_baselines = final_model.default_prediction() + final_model.compute_baselines()[
#         1]  # mean rating + item baseline ?
#
#     return pd.DataFrame({"nootropic": avalaible_nootropics, "item_baselines":item_baselines})

def interpret_prediction(trainset, model, avalaible_nootropics, user_id, predicted_ratings, rating_dic, item_baselines_user):
    # Disappointing results for now, try again with more data
    similarity_matrix = model.compute_similarities()
    print("ritalin adderall")
    print(similarity_matrix[trainset.to_inner_iid("Methylphenidate (Ritalin)"), trainset.to_inner_iid("Adderall")])
    #similarity_matrix -= np.eye(len(similarity_matrix))
    indices_available_nootropics = np.array([trainset.to_inner_iid(item) for item in avalaible_nootropics])
    similarity_matrix = similarity_matrix[indices_available_nootropics][:, indices_available_nootropics]

    for i, nootropic in enumerate(avalaible_nootropics):
        print("########")
        print("Nootropic: " + nootropic)
        predicted_ratings.append(model.predict(user_id, nootropic).est)
        similarities = similarity_matrix[np.where(indices_available_nootropics == trainset.to_inner_iid(nootropic))[0]].reshape(-1)
        #similarities = similarity_matrix[trainset.to_inner_iid(nootropic)].reshape(-1)
        importances = np.zeros(len(similarities))
        sim_sum = 0
        for item in rating_dic:
            if not rating_dic[item] is None and item != nootropic and item in avalaible_nootropics:
                indice_item = np.where(indices_available_nootropics == trainset.to_inner_iid(item))[0]
                #indice_item = trainset.to_inner_iid(item)
                #print(item)
                #print(rating_dic[item])
                #print(item_baselines_user[indice_item])
                sim_sum += similarities[indice_item]
                #print(np.abs(similarities[indice_item] * (rating_dic[item] - item_baselines_user[indice_item])))
                importances[indice_item] = similarities[indice_item]# * (rating_dic[item] - item_baselines_user[indice_item])
        importances = importances / sim_sum
        ind = np.argsort(np.abs(importances))[::-1][:20]
        print([trainset.to_raw_iid(i) for i in ind])
        print(importances[ind])
        print(item_baselines_user[np.where(indices_available_nootropics == trainset.to_inner_iid(nootropic))[0]])
        print(item_baselines_user[np.where(indices_available_nootropics == trainset.to_inner_iid(nootropic))[0]] + np.sum(importances))
        print(predicted_ratings[i])

def predict(rating_dic):
    df_clean = load_database()
    avalaible_nootropics = np.unique(df_clean["itemID"]) #we want to ignore nootropics that are not in the df
    #######################
    # Fit surprise model
    #######################

    #final_model = KNNBaseline(k=100, min_k=2, sim_options={'name': 'pearson_baseline', 'user_based': True})
    final_model = KNNBaseline(**{'verbose': False, 'k': 50, 'min_k': 5, #check
                                 'sim_options': {'name': 'msd', 'user_based': False},
                                 'bsl_options': {'method': 'sgd', 'n_epochs': 20}})
    #final_model = SVD(**{'n_factors': 10, 'n_epochs': 20, 'lr_all': 0.005, 'reg_all': 0.1})

    new_user_id = max(df_clean["userID"]) + 1 #TODO if merge
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
    for nootropic in avalaible_nootropics:
            predicted_ratings.append(final_model.predict(new_user_id, nootropic).est)



    #item_baselines_inner = final_model.default_prediction() + final_model.compute_baselines()[1]  # mean rating + item baseline ?
    #item_baselines_raw = []
    #convert inner to raw https://surprise.readthedocs.io/en/stable/FAQ.html#raw-inner-note
    #for nootropic in avalaible_nootropics:
    #    item_baselines_raw.append(item_baselines_inner[new_trainset.to_inner_iid(nootropic)])

    #item_baselines_raw = np.array(item_baselines_raw)
    #print(final_model.compute_baselines()[0][-1])
    #item_baselines_user = item_baselines_raw + final_model.compute_baselines()[0][-1]

    #interpret_prediction(new_trainset, final_model, avalaible_nootropics, new_user_id, predicted_ratings, rating_dic, item_baselines_user)
    #mean_boost = np.median((predicted_ratings - item_baselines_user) / item_baselines_user)


    mean_rating = [np.mean(df_clean[df_clean["itemID"] == noot]["rating"]) for noot in avalaible_nootropics]
    result_df = pd.DataFrame(
        {"nootropic": [noot for noot in avalaible_nootropics],
         "Prediction": predicted_ratings,
         "Mean rating": mean_rating})
         #"Boost": 100 * ((predicted_ratings - item_baselines_user) / item_baselines_user - mean_boost)})
    # "baseline_rating_user": item_baselines_user}) #TODO ?
    mask = [noot not in rating_dic.keys() for noot in avalaible_nootropics]
    result_df = result_df.iloc[mask]




    return result_df.sort_values("Prediction", ascending=False, ignore_index=True)


# def evaluate_2(new_user_id, final_model, rating_dic):
#     df_clean = load_database()
#     avalaible_nootropics = np.unique(df_clean["itemID"]) #we want to ignore nootropics that are not in the df
#     final_model.sim -= np.eye(len(final_model.sim)) #check if necesseray
#     predicted_ratings = []
#     old_baseline_user = final_model.compute_baselines()[0][-1]
#     rating_dic_copy = deepcopy(rating_dic)
#     rated_avalaible_nootropics = [nootropic for nootropic in rating_dic.keys() if nootropic in avalaible_nootropics]
#     for nootropic in rated_avalaible_nootropics[:10]:  # limit to 10 the number to prevent to much time to fit #TODO make it easier to compute
#         rating_dic_copy.pop(nootropic)
#         baseline_nootropic = 0
#
#         rating_dic_copy = deepcopy(rating_dic)
#     for nootropic in avalaible_nootropics:
#         loo_baseline_user = np.mean([rating - baseline_item for item in rating_dic.pop]) + reg / (len(rating_dic.pop))
#
#         predicted_ratings.append(final_model.predict(new_user_id, nootropic).est - old_baseline_user + loo_baseline_user)

def evaluate(rating_dic):
    df_clean = load_database()
    avalaible_nootropics = np.unique(df_clean["itemID"]) #we want to ignore nootropics that are not in the df
    loo_ratings = []
    rating_dic_copy = deepcopy(rating_dic)
    rated_avalaible_nootropics = [nootropic for nootropic in rating_dic.keys() if nootropic in avalaible_nootropics]
    rated_avalaible_nootropics = np.random.choice(rated_avalaible_nootropics, min(len(rated_avalaible_nootropics), 10), replace=False)
    if len(rated_avalaible_nootropics) < 2:
        st.warning("Please rate more nootropics")
        return None
    else:
        for nootropic in rated_avalaible_nootropics: #limit to 10 the number to prevent to much time to fit #TODO make it easier to compute
            rating_dic_copy.pop(nootropic)
            new_result_df = predict(rating_dic_copy)
            loo_ratings.append(new_result_df[new_result_df["nootropic"] == nootropic]["Prediction"].values[0])
            rating_dic_copy = deepcopy(rating_dic)

    #item_baselines_df = get_item_baseline()
    #item_baselines = item_baselines_df[item_baselines_df["nootropic"].isin(rating_dic.keys())]["item_baselines"].values
    mean_rating = [np.mean(df_clean[df_clean["itemID"] == noot]["rating"]) for noot in rated_avalaible_nootropics]


    return pd.DataFrame({"nootropic": [noot for noot in rated_avalaible_nootropics],
                         "Your rating": [rating_dic[nootropic] for nootropic in rated_avalaible_nootropics],
                         "Predicted rating": loo_ratings,
                         "Mean rating": mean_rating})

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