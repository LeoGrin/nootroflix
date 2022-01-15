import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from surprise import SlopeOne, CoClustering, KNNWithZScore, KNNWithMeans, SVDpp, BaselineOnly
import surprise
from surprise.model_selection import cross_validate, GridSearchCV
from surprise import Reader, Dataset, KNNBaseline, SVD, accuracy
from experiments.new_names import all_nootropics
import numpy as np
import json
from sklearn.metrics import mean_squared_error

OVERWRITE = False

## Algorithm selection

# %%

df_clean = pd.read_csv("data/total_df.csv")

# A reader is still needed but only the rating_scale param is requiered.
reader = Reader(rating_scale=(0, 10))

# The columns must correspond to user id, item id and ratings (in that order).
data = Dataset.load_from_df(df_clean, reader)

# %%


# We can now use this dataset as we please, e.g. calling cross_validate
ALL_ALGO = True
RANDOM_SEARCH = True


if ALL_ALGO:
    algorithms = ["SlopeOne",
                  "CoClustering",
                  "SVD",
                  "SVDpp",
                  "KNN_means_users",
                  "KNN_zscore_users",
                  "KNN_baselines_users",
                  "KNN_means_items",
                  "KNN_zscore_items",
                  "KNN_baselines_items",
                  "BaselineOnly"]
    rmse = []
    mae = []
    fcp = []
    print("SlopeOne")
    res = cross_validate(SlopeOne(), data, cv=5, measures = ["rmse", "mae", "fcp"], verbose=True)
    print(res)
    rmse.append(np.mean(res["test_rmse"]))
    mae.append(np.mean(res["test_mae"]))
    fcp.append(np.mean(res["test_fcp"]))
    print("CoClustering")
    res = cross_validate(CoClustering(), data, cv=5, measures = ["rmse", "mae", "fcp"], verbose=True)
    rmse.append(np.mean(res["test_rmse"]))
    mae.append(np.mean(res["test_mae"]))
    fcp.append(np.mean(res["test_fcp"]))
    print("SVD")
    res = cross_validate(SVD(), data, cv=5, measures = ["rmse", "mae", "fcp"], verbose=True)
    rmse.append(np.mean(res["test_rmse"]))
    mae.append(np.mean(res["test_mae"]))
    fcp.append(np.mean(res["test_fcp"]))
    print("SVDpp")
    res = cross_validate(SVDpp(), data, cv=5, measures = ["rmse", "mae", "fcp"], verbose=True)
    rmse.append(np.mean(res["test_rmse"]))
    mae.append(np.mean(res["test_mae"]))
    fcp.append(np.mean(res["test_fcp"]))
    print("KNN... (user based)")
    print("with means")
    res = cross_validate(KNNWithMeans(), data, cv=5, measures = ["rmse", "mae", "fcp"], verbose=True)
    rmse.append(np.mean(res["test_rmse"]))
    mae.append(np.mean(res["test_mae"]))
    fcp.append(np.mean(res["test_fcp"]))
    print("with z-score")
    res = cross_validate(KNNWithZScore(), data, cv=5, measures = ["rmse", "mae", "fcp"], verbose=True)
    rmse.append(np.mean(res["test_rmse"]))
    mae.append(np.mean(res["test_mae"]))
    fcp.append(np.mean(res["test_fcp"]))
    print("with baselines")
    res = cross_validate(KNNBaseline(), data, cv=5, measures = ["rmse", "mae", "fcp"], verbose=True)
    rmse.append(np.mean(res["test_rmse"]))
    mae.append(np.mean(res["test_mae"]))
    fcp.append(np.mean(res["test_fcp"]))
    print("KNN... (item based)")
    print("with means")
    res = cross_validate(KNNWithMeans(sim_options = {'user_based': False}), data, cv=5, measures = ["rmse", "mae", "fcp"], verbose=True)
    rmse.append(np.mean(res["test_rmse"]))
    mae.append(np.mean(res["test_mae"]))
    fcp.append(np.mean(res["test_fcp"]))
    print("with z-scores")
    res = cross_validate(KNNWithZScore(sim_options = {'user_based': False}), data, cv=5, measures = ["rmse", "mae", "fcp"], verbose=True)
    rmse.append(np.mean(res["test_rmse"]))
    mae.append(np.mean(res["test_mae"]))
    fcp.append(np.mean(res["test_fcp"]))
    print("with baselines")
    res = cross_validate(KNNBaseline(sim_options = {'user_based': False}), data, cv=5, measures = ["rmse", "mae", "fcp"], verbose=True)
    rmse.append(np.mean(res["test_rmse"]))
    mae.append(np.mean(res["test_mae"]))
    fcp.append(np.mean(res["test_fcp"]))
    print("BaselineOnly")
    res = cross_validate(BaselineOnly(), data, cv=5, measures = ["rmse", "mae", "fcp"], verbose=True)
    rmse.append(np.mean(res["test_rmse"]))
    mae.append(np.mean(res["test_mae"]))
    fcp.append(np.mean(res["test_fcp"]))

    # %%

    res_df = pd.DataFrame({"algo" :algorithms, "rmse" :rmse, "mae" :mae, "fcp" :fcp})
    print(res_df)

    if OVERWRITE:
        res_df.to_csv("model_selection/res.csv")

# %% md


# %% md

## Hyperparameters tuning

# %%
if RANDOM_SEARCH:
    svd_params_dic = {"verbose":[False],
                      "n_factors" :[10, 50, 100, 300], "n_epochs" :[20, 40, 100], "lr_all" :[0.005, 0.1], "reg_all" :[0.02, 0.1, 0.002]}

    param_search = GridSearchCV(SVD, svd_params_dic, cv=5, n_jobs=-1)
    param_search.fit(data)

    # %%

    print(param_search.best_params)
    print(param_search.best_score)

    # %%

    knn_params_dic = {"verbose":[False],
                      "k" :[10, 20, 40, 60, 100],
                      "min_k" :[1, 2, 5, 10],
                      "sim_options" :{'name': ['pearson_baseline', 'msd', 'cosine'], "user_based" :[True, False]}}

    knn_param_search = GridSearchCV(KNNBaseline, knn_params_dic, cv=5, n_jobs=-1)
    knn_param_search.fit(data)


    print(knn_param_search.best_params)
    print(knn_param_search.best_score)

knn1 = KNNBaseline(**{'verbose': False, 'k': 100, 'min_k': 5, 'sim_options': {'name': 'msd', 'user_based': False}})
knn2 = KNNBaseline(k=60,
                           min_k=2,
                           verbose=False,
                           sim_options={'name': 'pearson_baseline', 'user_based': True})
knn3 = SVD(**{'n_factors': 50, 'n_epochs': 20, 'lr_all': 0.005, 'reg_all': 0.1})

cross_validate(knn1,
               data,
               measures = ["rmse", "mae", "fcp"],
               verbose=True)
cross_validate(knn2,
               data,
               measures=["rmse", "mae", "fcp"],
               verbose=True)

cross_validate(knn3,
               data,
               measures = ["rmse", "mae", "fcp"],
               verbose=True)

kSplit = surprise.model_selection.split.KFold(n_splits=10, shuffle=True) # split data into folds.

 #try removing sim_options. You'll find memory errors.
rmse = []
stacking_regressor = RandomForestRegressor()
for i in range(30):
    train_val_indices = np.random.choice(list(range(len(df_clean))), int(len(df_clean) * 0.8), replace=False)
    train_indices, val_indices = train_val_indices[:int(len(train_val_indices) * 0.5)], train_val_indices[int(len(train_val_indices) * 0.5):]
    test_indices = np.array([i for i in range(len(df_clean)) if i not in train_val_indices])
    print(len(train_indices))
    print(len(val_indices))
    print(len(test_indices))

    df_train, df_val, df_test = df_clean.iloc[train_indices], df_clean.iloc[val_indices], df_clean.iloc[test_indices]
    reader = Reader(rating_scale=(0, 10))
    # The columns must correspond to user id, item id and ratings (in that order).
    trainset = Dataset.load_from_df(df_train, reader).build_full_trainset()
    valset = Dataset.load_from_df(df_val, reader).build_full_trainset().build_testset()
    val_ratings = df_val["rating"].values
    test_ratings = df_test["rating"].values
    testset = Dataset.load_from_df(df_test, reader).build_full_trainset().build_testset()
    knn1.fit(trainset)
    knn2.fit(trainset)
    knn3.fit(trainset)
    predictions_val_1 = np.array([pred.est for pred in knn1.test(valset)]).reshape(-1, 1)
    predictions_val_2 = np.array([pred.est for pred in knn2.test(valset)]).reshape(-1, 1)
    predictions_val_3 = np.array([pred.est for pred in knn3.test(valset)]).reshape(-1, 1)
    predictions_test_1 = np.array([pred.est for pred in knn1.test(testset)]).reshape(-1, 1)
    predictions_test_2 = np.array([pred.est for pred in knn2.test(testset)]).reshape(-1, 1)
    predictions_test_3 = np.array([pred.est for pred in knn3.test(testset)]).reshape(-1, 1)
    stacking_regressor.fit(np.concatenate([predictions_val_1, predictions_val_2, predictions_val_3], axis=1), val_ratings)
    stacking_predictions = stacking_regressor.predict(np.concatenate([predictions_test_1, predictions_test_2, predictions_test_3], axis=1))
    rmse.append(np.sqrt(mean_squared_error(test_ratings, stacking_predictions)))
    print(np.sqrt(mean_squared_error(test_ratings, stacking_predictions)))
    print(np.sqrt(mean_squared_error(test_ratings, predictions_test_1)))
    print(np.sqrt(mean_squared_error(test_ratings, predictions_test_2)))
    print(np.sqrt(mean_squared_error(test_ratings, predictions_test_3)))
    print(np.sqrt(mean_squared_error(test_ratings, np.mean(np.concatenate([predictions_test_1, predictions_test_2, predictions_test_3], axis=1), axis=1))))
    #df_test = df_test[df_test["itemID"].isin(nootropics_with_enough_ratings)]
print(np.mean(rmse))

if OVERWRITE:
    with open("model_selection/scores.txt", "w") as f:
        f.write(json.dumps(knn_param_search.best_params))
        f.write("\n")
        f.write(json.dumps(knn_param_search.best_score))
        f.write("\n")
        f.write(json.dumps(param_search.best_params))
        f.write("\n")
        f.write(json.dumps(param_search.best_score))

#Test models on original data

def evaluate(df_train, df_test, suffix=""):
    suprise_model_1 = KNNBaseline(k=60, min_k=2, verbose=False, sim_options={'name': 'pearson_baseline', 'user_based': True})
    suprise_model_2 = KNNBaseline(**{'verbose': False, 'k': 100, 'min_k': 5, 'sim_options': {'name': 'msd', 'user_based': False}})
    #suprise_model_2 = SVD(**{'n_factors': 50, 'n_epochs': 20, 'lr_all': 0.005, 'reg_all': 0.1})

    reader = Reader(rating_scale=(0, 10))
    # The columns must correspond to user id, item id and ratings (in that order).
    trainset = Dataset.load_from_df(df_train, reader).build_full_trainset()
    testset = Dataset.load_from_df(df_test, reader).build_full_trainset().build_testset()

    suprise_model_1.fit(trainset)
    suprise_model_2.fit(trainset)
    dic = {"model":[], "rmse":[], "mae":[], "fcp":[]}
    dic["model"].append("KNN_{}".format(suffix))
    print("Model 1")
    dic["rmse"].append(accuracy.rmse(suprise_model_1.test(testset)))
    dic["mae"].append(accuracy.mae(suprise_model_1.test(testset)))
    dic["fcp"].append(accuracy.fcp(suprise_model_1.test(testset)))

    dic["model"].append("SVD_{}".format(suffix))
    print("Model 2")
    dic["rmse"].append(accuracy.rmse(suprise_model_2.test(testset)))
    dic["mae"].append(accuracy.mae(suprise_model_2.test(testset)))
    dic["fcp"].append(accuracy.fcp(suprise_model_2.test(testset)))

    return pd.DataFrame(dic)

evaluate = False
if evaluate:
    #Test on nootropics with enough ratings
    print("Test on nootropics with enough ratings")
    nootropics_with_enough_ratings = []
    for noot in all_nootropics:
        if df_clean[df_clean["itemID"] == noot].shape[0] > 30:
            nootropics_with_enough_ratings.append(noot)

    df = pd.DataFrame()

    for i in range(30):
        train_indices = np.random.choice(list(range(len(df_clean))), int(len(df_clean) * 0.6), replace=False)
        test_indices = np.array([i for i in range(len(df_clean)) if i not in train_indices])

        df_train, df_test = df_clean.iloc[train_indices], df_clean.iloc[test_indices]

        df_test = df_test[df_test["itemID"].isin(nootropics_with_enough_ratings)]

        df = df.append(evaluate(df_train, df_test))

    print(df.groupby("model").agg([np.mean, np.std]))


    # test on original data
    print("TESTING ON ORIGINAL DATA")
    df_ssc = pd.read_csv("data/dataset_clean_right_names.csv")
    df_new = pd.read_csv("data/new_df.csv")

    remove_rare_noot = False

    if remove_rare_noot:
        nootropics_with_enough_ratings = []
        for noot in all_nootropics:
            if df_new[df_new["itemID"] == noot].shape[0] > 10:  # TODO : optimize the number
                nootropics_with_enough_ratings.append(noot)
        df_train_total = df_new[df_new["itemID"].isin(nootropics_with_enough_ratings)]

    n_ratings_ssc = len(df_ssc)
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()

    for i in range(30):
        train_indices = np.random.choice(list(range(n_ratings_ssc)), int(n_ratings_ssc * 0.6), replace=False)
        test_indices = np.array([i for i in range(n_ratings_ssc) if i not in train_indices])

        df_ssc_train, df_ssc_test = df_ssc.iloc[train_indices], df_ssc.iloc[test_indices]

        df1 = df1.append(evaluate(df_ssc_train, df_ssc_test, "ssc"))

        df_train_total = pd.concat([df_new, df_ssc_train])

        remove_rare_noot = False

        df2 = df2.append(evaluate(df_train_total, df_ssc_test, "new"))


    df_total = df1.append(df2)
    print(df_total.groupby("model").agg([np.mean, np.std]))

    if OVERWRITE:
        df_total.groupby("model").agg([np.mean, np.std]).to_csv("model_selection/score_on_original_mean.csv")

