import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from data.code import code
import ot

# We have three databases with different scales.
# We merge then by computing the OT transport between the distributions
# A possible improvement: try to match the distribution of ratings for each nootropics
# mean mapping

df_darkha_original = pd.read_csv("../data/nootropics_survey_darkha.csv")
df_darkha_original = df_darkha_original[list(code.keys())]
df_darkha_original = df_darkha_original.rename(columns = code)
df_darkha_original = df_darkha_original.applymap(lambda x: x-1)


df_ssc = pd.read_csv("dataset_clean_features.csv")

darkha_features = set(code.values())
ssc_features = set(np.unique(df_ssc["itemID"]))
print("ssc features")
print(ssc_features)
intersection_features = list(ssc_features.intersection(darkha_features))
print(intersection_features)
print("ok")
print(ssc_features.difference(intersection_features))

df_darkha = df_darkha_original[intersection_features]
df_ssc = df_ssc[np.isin(df_ssc["itemID"], intersection_features)]


plt.hist(df_ssc["rating"])
plt.show()


#OT

# 1st method: match the total distribution of ratings

#equalize the proportion of rating for each nootripc so that we can match the total distribution
# n_ratings_darkha = sum(~pd.isnull(df_darkha).values.reshape(-1))
# new_rating_ssc = []
# for item in intersection_features:
#     ssc_ratings_item = df_ssc[df_ssc["itemID"] == item]["rating"]
#     new_rating_ssc.extend(list(np.random.choice(ssc_ratings_item, int(len(df_ssc) * sum(~np.isnan(df_darkha[item])) / n_ratings_darkha), replace=True)))
#
# new_rating_ssc = np.array(new_rating_ssc)


# ratings_darkha = df_darkha.values.reshape(-1)
# ratings_darkha = ratings_darkha[~np.isnan(ratings_darkha)]
#
#
# n_ssc_ratings = len(new_rating_ssc)
# _,counts = np.unique(new_rating_ssc, return_counts=True)
# ssc_hist = counts / n_ssc_ratings
# print(ssc_hist)
# print(sum(ssc_hist))
#
# _, counts = np.unique(ratings_darkha, return_counts=True)
# darkha_hist = counts / len(ratings_darkha)
# print(darkha_hist)
# print(sum(darkha_hist))
#
# M = ot.dist(np.array(range(len(darkha_hist))).reshape(-1, 1), np.array(range(len(ssc_hist))).reshape(-1, 1))
# M = M / M.max()
#
# T = ot.emd(darkha_hist, ssc_hist, M) #coupling matrix
#
# def conversion(rating_original, coupling):
#     if np.isnan(rating_original):
#         return np.nan
#     rating_original = int(rating_original)
#     probas = coupling[rating_original] / sum(coupling[rating_original])
#     new_rating = np.random.choice(range(len(probas)), size=1, p=probas)[0]
#     return new_rating

# 2nd method: match the distribution of ratings for each nootropics, if we have enough samples
for nootropic in intersection_features:
    ratings_darkha = df_darkha[nootropic].values
    ratings_darkha = ratings_darkha[~np.isnan(ratings_darkha)]
    print(len(ratings_darkha))

    ratings_ssc = df_ssc[df_ssc["itemID"] == nootropic]["rating"]
    n_ssc_ratings = len(ratings_ssc)
    print(n_ssc_ratings)
    print("########")
    _, counts = np.unique(ratings_ssc, return_counts=True)
    ssc_hist = counts / n_ssc_ratings
    #print(ssc_hist)
    #print(sum(ssc_hist))

    #val, counts = np.unique(ratings_darkha, return_counts=True)
    counts = []
    for i in range(7):
        counts.append(sum(ratings_darkha == i))
    darkha_hist = np.array(counts) / len(ratings_darkha)
   # print(darkha_hist)
    #print(sum(darkha_hist))

    M = ot.dist(np.array(range(len(darkha_hist))).reshape(-1, 1), np.array(range(len(ssc_hist))).reshape(-1, 1))
    M = M / M.max()

    T = ot.emd(darkha_hist, ssc_hist, M) #coupling matrix

    def conversion(rating_original, coupling):
        if np.isnan(rating_original):
            return np.nan
        rating_original = int(rating_original)
        probas = coupling[rating_original] / sum(coupling[rating_original])
        new_rating = np.random.choice(range(len(probas)), size=1, p=probas)[0]
        return new_rating
    df_darkha_original[nootropic] = list(map(lambda x: conversion(x, T), df_darkha_original[nootropic]))

df_darkha_original.to_csv("../data/nootropics_survey_darkha_converted_2.csv", index=False)

converted_ratings_darkha = []
for rating in (ratings_darkha.astype(int)):
    probas = T[rating] / sum(T[rating])
    new_rating = np.random.choice(range(len(probas)), size=1, p=probas)[0]
    converted_ratings_darkha.append(new_rating)

#plt.hist(converted_ratings_darkha)
#plt.show()
#plt.hist(df_ssc["rating"])
#plt.show()
# ratings_darkha_caffeine = df_darkha["Noopept"]
# ratings_darkha_caffeine = ratings_darkha_caffeine[~np.isnan(ratings_darkha_caffeine)]
#
# converted_ratings_darkha = []
# for rating in (ratings_darkha_caffeine.astype(int)):
#     probas = T[rating] / sum(T[rating])
#     new_rating = np.random.choice(range(len(probas)), size=1, p=probas)[0]
#     converted_ratings_darkha.append(new_rating)
# #%%
# fig, axs = plt.subplots(2)
# axs[0].hist(converted_ratings_darkha)
# #%%
# axs[1].hist(df_ssc[df_ssc["itemID"] == "Noopept"]["rating"])
#
# plt.show()
#
#
