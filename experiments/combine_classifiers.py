import sklearn.experimental.enable_hist_gradient_boosting
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingClassifier, GradientBoostingRegressor
import pandas as pd
from sklearn.model_selection import cross_validate, train_test_split
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import svm

from surprise import NormalPredictor, SVD, NMF, SlopeOne, CoClustering, KNNBasic, KNNWithZScore, KNNWithMeans, KNNBaseline, SVDpp
from surprise import Dataset
from surprise import Reader
#from surprise.model_selection import cross_validate, RandomizedSearchCV


df = pd.read_csv('dataset_clean_features.csv')

def compute_metrics(classifier=GradientBoostingRegressor, stacking_classifier=RandomForestRegressor, prop_val = 0.3):
    res_dic = {}

    indices = np.array(list(range(len(df))))
    np.random.shuffle(indices)
    train_indices, val_indices, test_indices = indices[:int(0.5 * len(indices))], indices[int(0.5 * len(indices)):int(
        (0.5 + prop_val) * len(indices))], indices[int((0.5 + prop_val) * len(indices)):]

    ct = ColumnTransformer(
        [("categorical", OneHotEncoder(sparse=False), ['itemID', 'MentalHealthAnxiety', "Age", 'MentalHealthDepression', 'MentalHealthADHD', 'Sex'])])

    X = ct.fit_transform(df.drop(["CountryofResidence", "rating", "userID"], axis=1))

    y = df["rating"]
    X_train, X_test, X_val, y_train, y_test, y_val = X[train_indices], \
                                                     X[test_indices], \
                                                     X[val_indices], \
                                                     y[train_indices], \
                                                     y[test_indices], \
                                                     y[val_indices]


    clf = classifier

    clf.fit(X_train, y_train)
    #print("clf error on test")
    #print(mean_squared_error(y_test, clf.predict(X_test), squared=False))
    #print(mean_absolute_error(y_test, clf.predict(X_test)))
    res_dic["classifier_rmse"] = mean_squared_error(y_test, clf.predict(X_test), squared=False)
    res_dic["classifier_mae"] = mean_absolute_error(y_test, clf.predict(X_test))

    df_surprise_train = df[["userID", "itemID", "rating"]].iloc[train_indices]
    df_surprise_test = df[["userID", "itemID", "rating"]].iloc[test_indices]
    df_surprise_val = df[["userID", "itemID", "rating"]].iloc[val_indices]
    df_surprise_train_val = df[["userID", "itemID", "rating"]].iloc[list(train_indices) + list(val_indices)]


    suprise_model_1 = KNNBaseline(k=60, min_k=2, sim_options={'name': 'pearson_baseline', 'user_based': True})
    suprise_model_2 = SVD(**{'n_factors': 50, 'n_epochs': 20, 'lr_all': 0.005, 'reg_all': 0.1})


    # A reader is still needed but only the rating_scale param is requiered.
    reader = Reader(rating_scale=(0, 10))

    # The columns must correspond to user id, item id and ratings (in that order).
    new_trainset = Dataset.load_from_df(df_surprise_train, reader).build_full_trainset()

    suprise_model_1.fit(new_trainset)
    suprise_model_2.fit(new_trainset)
    prediction_surprise_1 = lambda row: suprise_model_1.predict(uid=row["userID"], iid=row["itemID"]).est
    prediction_surprise_2 = lambda row: suprise_model_2.predict(uid=row["userID"], iid=row["itemID"]).est
    val_predictions_surprise_1 = df_surprise_val.apply(prediction_surprise_1, axis=1)
    test_predictions_surprise_1 = df_surprise_test.apply(prediction_surprise_1, axis=1)
    val_predictions_surprise_2 = df_surprise_val.apply(prediction_surprise_2, axis=1)
    test_predictions_surprise_2 = df_surprise_test.apply(prediction_surprise_2, axis=1)
    val_predictions_clf = clf.predict(X_val)
    test_predictions_clf = clf.predict(X_test)

    X_val_both = np.concatenate((val_predictions_surprise_1.values.reshape(-1, 1), val_predictions_surprise_2.values.reshape(-1, 1)), axis=1)
    X_test_both = np.concatenate((test_predictions_surprise_1.values.reshape(-1, 1), test_predictions_surprise_2.values.reshape(-1, 1)), axis=1)
    clf_stacking = stacking_classifier()
    clf_stacking.fit(X_val_both, y_val)
    #print(mean_squared_error(y_test, clf_stacking.predict(X_test), squared=False))
    #print(mean_absolute_error(y_test, clf_stacking.predict(X_test)))
    res_dic["stacking_both_rmse"] = mean_squared_error(y_test, clf_stacking.predict(X_test_both), squared=False)
    res_dic["stacking_both_mae"] = mean_absolute_error(y_test, clf_stacking.predict(X_test_both))


    #clf.fit(X_val, y_val)
    #print("clf error on test (trained on val)")
    #print(mean_squared_error(y_test, clf.predict(X_test), squared=False))
    #print(mean_absolute_error(y_test, clf.predict(X_test)))
    #res_dic["clf_trained_val_rmse"] = mean_squared_error(y_test, clf.predict(X_test), squared=False)
    #res_dic["clf_trained_val_mae"] = mean_absolute_error(y_test, clf.predict(X_test))
    # clf.fit(X_val, y_val)
    # #print("clf error on test (trained on val with suprise predictions)")
    # #print(mean_squared_error(y_test, clf.predict(X_test), squared=False))
    # #print(mean_absolute_error(y_test, clf.predict(X_test)))
    # res_dic["clf_trained_val_suprise_rmse"] = mean_squared_error(y_test, clf.predict(X_test), squared=False)
    # res_dic["clf_trained_val_suprise_mae"] = mean_absolute_error(y_test, clf.predict(X_test))
    #
    # X_val = val_predictions_surprise.values.reshape(-1, 1)
    # X_test = test_predictions_surprise.values.reshape(-1, 1)
    #
    #
    # clf.fit(X_val, y_val)
    # #print("clf error on test (trained on val with suprise predictions)")
    # #print(mean_squared_error(y_test, clf.predict(X_test), squared=False))
    # #print(mean_absolute_error(y_test, clf.predict(X_test)))
    # res_dic["clf_trained_val_only_surprise_rmse"] = mean_squared_error(y_test, clf.predict(X_test), squared=False)
    # res_dic["clf_trained_val_only_surprise_mae"] = mean_absolute_error(y_test, clf.predict(X_test))

    #print("surprise error on test")
    #print(mean_squared_error(y_test, test_predictions_surprise, squared=False))
    #print(mean_absolute_error(y_test, test_predictions_surprise))
    res_dic["surprise_KNN_test_rmse"] = mean_squared_error(y_test, test_predictions_surprise_1, squared=False)
    res_dic["surprise_KNN_test_mae"] = mean_absolute_error(y_test, test_predictions_surprise_1)

    res_dic["surprise_SVD_test_rmse"] = mean_squared_error(y_test, test_predictions_surprise_2, squared=False)
    res_dic["surprise_SVD_test_mae"] = mean_absolute_error(y_test, test_predictions_surprise_2)

    final_model_train_val = KNNBaseline(k=60, min_k=2, sim_options={'name': 'pearson_baseline', 'user_based': True})

    # A reader is still needed but only the rating_scale param is requiered.
    reader = Reader(rating_scale=(0, 10))

    # The columns must correspond to user id, item id and ratings (in that order).
    new_trainset = Dataset.load_from_df(df_surprise_train_val, reader).build_full_trainset()

    final_model_train_val.fit(new_trainset)

    prediction_train_val = lambda row: final_model_train_val.predict(uid=row["userID"], iid=row["itemID"]).est
    test_predictions_surprise_train_val = df_surprise_test.apply(prediction_train_val, axis=1)
    res_dic["surprise_test_rmse_train_val"] = mean_squared_error(y_test, test_predictions_surprise_train_val, squared=False)
    res_dic["surprise_test_mae_train_val"] = mean_absolute_error(y_test, test_predictions_surprise_train_val)




    #print("stacking predictions (using all features)")
    X_val_features_KNN = np.concatenate((val_predictions_surprise_1, val_predictions_clf.reshape(-1, 1)), axis=1)
    X_test_features_KNN = np.concatenate((X_test, test_predictions_clf.reshape(-1, 1)), axis=1)
    clf_stacking = stacking_classifier()
    clf_stacking.fit(X_val_features_KNN, y_val)
    #print(mean_squared_error(y_test, clf_stacking.predict(X_test), squared=False))
    #print(mean_absolute_error(y_test, clf_stacking.predict(X_test)))
    res_dic["stacking_features_KNN_rmse"] = mean_squared_error(y_test, clf_stacking.predict(X_test_features_KNN), squared=False)
    res_dic["stacking_features_KNN_mae"] = mean_absolute_error(y_test, clf_stacking.predict(X_test_features_KNN))

    X_val_features_SVD = np.concatenate((val_predictions_surprise_2, val_predictions_clf.reshape(-1, 1)), axis=1)
    X_test_features_SVD = np.concatenate((X_test, test_predictions_clf.reshape(-1, 1)), axis=1)
    clf_stacking = stacking_classifier()
    clf_stacking.fit(X_val_features_SVD, y_val)
    #print(mean_squared_error(y_test, clf_stacking.predict(X_test), squared=False))
    #print(mean_absolute_error(y_test, clf_stacking.predict(X_test)))
    res_dic["stacking_features_SVD_rmse"] = mean_squared_error(y_test, clf_stacking.predict(X_test_features_SVD), squared=False)
    res_dic["stacking_features_SVD_mae"] = mean_absolute_error(y_test, clf_stacking.predict(X_test_features_SVD))


    return res_dic

    #TODO: cross validate for surprise prediction
    #predict(df.drop(["Age", 'CountryofResidence', 'Sex', 'MentalHealthDepression',
#       'MentalHealthAnxiety', 'MentalHealthADHD'], axis=1))


if __name__ == "__main__":
    res_df = pd.DataFrame()
    classifier = RandomForestRegressor(**{'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 100}) #best param #TODO automate
    #classifier = DummyRegressor()
    n = 20
    for _ in range(20):
        res_dic = compute_metrics(classifier=classifier)
        res_df = res_df.append(res_dic, ignore_index=True)
    new_df = pd.DataFrame({"mean": res_df.mean(),
                           "CI-": res_df.mean() - 2 * res_df.std() / np.sqrt(n),
                           "CI+":res_df.mean() + 2*res_df.std() /np.sqrt(n)})
    print(new_df)
    print(new_df.T)
