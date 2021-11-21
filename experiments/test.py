import pandas as pd
import sklearn.model_selection
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from surprise import Reader, Dataset, KNNBaseline, SVD, accuracy
import numpy as np

df_ssc = pd.read_csv("../data/dataset_clean.csv")
df_reddit = pd.read_csv("../data/nootropics_survey_reddit_converted_2.csv")
df_darkha = pd.read_csv("../data/nootropics_survey_darkha_converted_2.csv")

def reformat_dataframe(df):
    """Make it compatible with Surprise"""
    # transform into a 3 columns (user_id, item_id, rating) matrix for Surprise
    df = df.rename_axis('userID').reset_index()
    df = df.melt(id_vars="userID", var_name="itemID", value_name="rating")
    # remove rows when there is no rating
    df = df[~pd.isnull(df["rating"])]
    df = df.reset_index(drop=True)
    return df

df_reddit = reformat_dataframe(df_reddit)
df_darkha = reformat_dataframe(df_darkha)
names = ["ssc", "reddit", "darha"]
dataframes = [df_ssc, df_reddit, df_darkha]
for i, df in enumerate(dataframes):
    df["userID"] = names[i] + df["userID"].astype(str)

n_ratings_ssc = len(df_ssc)
train_indices = np.random.choice(list(range(n_ratings_ssc)), int(n_ratings_ssc * 0.7), replace=False)
test_indices = np.array([i for i in range(n_ratings_ssc) if i not in train_indices])

df_ssc_train, df_ssc_test = df_ssc.iloc[train_indices], df_ssc.iloc[test_indices]

def evaluate(df_train):
    suprise_model_1 = KNNBaseline(k=60, min_k=2, sim_options={'name': 'pearson_baseline', 'user_based': True})
    suprise_model_2 = SVD(**{'n_factors': 50, 'n_epochs': 20, 'lr_all': 0.005, 'reg_all': 0.1})

    reader = Reader(rating_scale=(0, 10))
    # The columns must correspond to user id, item id and ratings (in that order).
    trainset = Dataset.load_from_df(df_train, reader).build_full_trainset()
    testset = Dataset.load_from_df(df_ssc_test, reader).build_full_trainset().build_testset()


    suprise_model_1.fit(trainset)
    suprise_model_2.fit(trainset)

    print("Model 1")
    accuracy.rmse(suprise_model_1.test(testset))
    accuracy.mae(suprise_model_1.test(testset))
    accuracy.fcp(suprise_model_1.test(testset))

    print("Model 2")
    accuracy.rmse(suprise_model_2.test(testset))
    accuracy.mae(suprise_model_2.test(testset))
    accuracy.fcp(suprise_model_2.test(testset))

def stacking(df_train_list):
    #suprise_model_1 = KNNBaseline(k=60, min_k=2, sim_options={'name': 'pearson_baseline', 'user_based': True})
    #suprise_model_2 = SVD(**{'n_factors': 50, 'n_epochs': 20, 'lr_all': 0.005, 'reg_all': 0.1})

    reader = Reader(rating_scale=(0, 10))
    valset = Dataset.load_from_df(df_ssc_val, reader).build_full_trainset().build_testset()
    uid_val, iid_val, ratings_val = zip(*valset)
    testset = Dataset.load_from_df(df_ssc_test, reader).build_full_trainset().build_testset()
    uid_test, iid_test, ratings_test = zip(*testset)

    predictor_list = []
    for i, df_train in enumerate(df_train_list):
        predictor_list.append(KNNBaseline(k=60, min_k=2, sim_options={'name': 'pearson_baseline', 'user_based': True}))
        # The columns must correspond to user id, item id and ratings (in that order).
        trainset = Dataset.load_from_df(df_train, reader).build_full_trainset()
        predictor_list[i].fit(trainset)

    predictions_val = np.concatenate([np.array(list(map(lambda x: predictor.predict(*x).est, zip(uid_val, iid_val)))).reshape(-1, 1) for predictor in predictor_list], axis=1)
    print(predictions_val.shape)
    print(predictions_val)
    stacking_clf = GradientBoostingRegressor()
    stacking_clf.fit(predictions_val, ratings_val)

    predictions_test = np.concatenate([np.array(list(map(lambda x: predictor.predict(*x).est, zip(uid_test, iid_test)))).reshape(-1, 1) for predictor in predictor_list], axis=1)
    predictions_test_stacking = stacking_clf.predict(predictions_test)
    print(mean_squared_error(predictions_test_stacking, ratings_test, squared=False))

    print([mean_squared_error(predictions_test[:, i], ratings_test, squared=False) for i in range(predictions_test.shape[1])])




print("all")
evaluate(pd.concat([df_reddit, df_darkha, df_ssc_train]))
print("darkha ssc ")
evaluate(pd.concat([df_darkha, df_ssc_train]))
print("reddit ssc ")
evaluate(pd.concat([df_reddit, df_ssc_train]))
print("ssc")
evaluate(df_ssc_train)


prop_val = 0.1
indices = np.array(list(range(len(df_ssc))))
np.random.shuffle(indices)
train_indices, val_indices, test_indices = indices[:int(0.5 * len(indices))], indices[int(0.5 * len(indices)):int(
    (0.5 + prop_val) * len(indices))], indices[int((0.5 + prop_val) * len(indices)):]
df_ssc_train = df_ssc[["userID", "itemID", "rating"]].iloc[train_indices]
df_ssc_test = df_ssc[["userID", "itemID", "rating"]].iloc[test_indices]
df_ssc_val = df_ssc[["userID", "itemID", "rating"]].iloc[val_indices]
df_ssc_train_val = df_ssc[["userID", "itemID", "rating"]].iloc[list(train_indices) + list(val_indices)]



print(stacking([df_ssc_train, df_reddit, df_darkha]))