import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from data.code import code
import ot

df_reddit_original = pd.read_csv("../data/nootropics_survey_reddit.csv").drop("Timestamp", axis=1)
#print(df_reddit_original.columns)
#ratings = df_reddit_original.values.reshape(-1)
#ratings = ratings[~pd.isnull(ratings)]
def word_to_rating(word):
    if word == "Yes, Negative":
        return 0
    elif word == "Yes, Indifferent":
        return 1
    elif word == "Yes, Positive":
        return 2
    else:
        return np.nan


df_reddit_original = df_reddit_original.applymap(word_to_rating)

df_reddit_original.columns = ['5-HTP', 'ALCAR', 'Alpha-GPC', 'Ashwagandha', 'Bacopa',
       'Black Seed Oil', 'Caffeine', 'CBD', 'Cerebrolysin', 'Coluracetam',
       'Seligiline', 'Dextroamphetamine (Speed)', 'Omega-3 Supplements',
       'Dihexa', 'Etifoxine', 'Gingko Biloba', 'Ginseng', 'Huperzine A',
       'IDRA-21', 'Inositol', 'Kratom', "Lion's Mane", "Theanine",
       'LSD', 'Magnesium', 'MAOI', 'Melatonin', 'Modafinil',
       'N-acetyl Cysteine (NAC)', 'N-methyl-cyclazodone', 'Nicotine',
       'Noopept', 'NSI-189', 'Oxiracetam', 'P21', 'Palmitoylethano',
       'Phenibut', 'Phenylpiracetam', 'Phosphatidyl Serine', 'Piracetam',
       'Pregenolone', 'PRL853', 'Psilocybin Microdose', 'rgpu-95',
       'Rhodiola', 'Ritalin LA', 'SemaxandNASemaxetc', 'Tryptophan', 'Tyrosine',
       'Uridine', 'Valerian Root']



df_ssc = pd.read_csv("dataset_clean_features.csv")


reddit_features = set(df_reddit_original.columns)
ssc_features = set(np.unique(df_ssc["itemID"]))
intersection_features = list(ssc_features.intersection(reddit_features))
print(intersection_features)


df_reddit = df_reddit_original[intersection_features]
df_ssc = df_ssc[np.isin(df_ssc["itemID"], intersection_features)]


#equalize the proportion of rating for each nootripc so that we can match the total distribution
# n_ratings_reddit = sum(~pd.isnull(df_reddit).values.reshape(-1))
# new_rating_ssc = []
# for item in intersection_features:
#     ssc_ratings_item = df_ssc[df_ssc["itemID"] == item]["rating"]
#     new_rating_ssc.extend(list(np.random.choice(ssc_ratings_item, int(len(df_ssc) * sum(~np.isnan(df_reddit[item])) / n_ratings_reddit), replace=True)))
#
# new_rating_ssc = np.array(new_rating_ssc)
#
#
# ratings_reddit = df_reddit.values.reshape(-1)
# ratings_reddit = ratings_reddit[~np.isnan(ratings_reddit)]
#
#
# n_ssc_ratings = len(new_rating_ssc)
# _,counts = np.unique(new_rating_ssc, return_counts=True)
# ssc_hist = counts / n_ssc_ratings
# print(ssc_hist)
# print(sum(ssc_hist))
#
# _, counts = np.unique(ratings_reddit, return_counts=True)
# darkha_hist = counts / len(ratings_reddit)
# print(darkha_hist)
# print(sum(darkha_hist))
#
# #OT
#
# # loss matrix
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
#
# df_reddit_original.applymap(lambda x:conversion(x, T)).to_csv("../data/nootropics_survey_reddit_converted.csv", index=False)


# 2nd method: match the distribution of ratings for each nootropics, if we have enough samples
for nootropic in intersection_features:
    ratings_reddit = df_reddit[nootropic].values
    ratings_reddit = ratings_reddit[~np.isnan(ratings_reddit)]
    print(len(ratings_reddit))

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
    for i in range(3):
        counts.append(sum(ratings_reddit == i))
    darkha_hist = np.array(counts) / len(ratings_reddit)
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
    df_reddit_original[nootropic] = list(map(lambda x: conversion(x, T), df_reddit_original[nootropic]))

df_reddit_original.to_csv("../data/nootropics_survey_reddit_converted_2.csv", index=False)


converted_ratings_darkha = []
for rating in (ratings_reddit.astype(int)):
    probas = T[rating] / sum(T[rating])
    new_rating = np.random.choice(range(len(probas)), size=1, p=probas)[0]
    converted_ratings_darkha.append(new_rating)

#plt.hist(converted_ratings_darkha)
#plt.show()
#plt.hist(df_ssc["rating"])
#plt.show()
# ratings_reddit_caffeine = df_reddit_original["Phenibut"]
# ratings_reddit_caffeine = ratings_reddit_caffeine[~np.isnan(ratings_reddit_caffeine)]
#
# converted_ratings_reddit = []
# for rating in (ratings_reddit_caffeine.astype(int)):
#     probas = T[rating] / sum(T[rating])
#     new_rating = np.random.choice(range(len(probas)), size=1, p=probas)[0]
#     converted_ratings_reddit.append(new_rating)
# #%%
# fig, axs = plt.subplots(2)
# axs[0].hist(converted_ratings_reddit)
# #%%
# axs[1].hist(df_ssc[df_ssc["itemID"] == "Phenibut"]["rating"])
#
# plt.show()
#
#
#



