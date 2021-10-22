import sklearn.experimental.enable_hist_gradient_boosting
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingClassifier, GradientBoostingRegressor
import pandas as pd
from sklearn.model_selection import cross_validate, RandomizedSearchCV, train_test_split
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt


df = pd.read_csv('experiments/dataset_clean_features.csv').drop(['userID', 'CountryofResidence'], axis=1)


clf = RandomForestRegressor()
#clf = GradientBoostingRegressor()

numerical_columns = ["Age"]
categorical_columns = ['Sex', 'MentalHealthDepression',
       'MentalHealthAnxiety', 'MentalHealthADHD', 'itemID']

ct = ColumnTransformer(
    [("categorical", OneHotEncoder(sparse=False), categorical_columns)])

#pipeline = Pipeline([("transfomer", ct), ("classifier", clf)])
#careful data leakage if I scale

X = ct.fit_transform(df.drop("rating", axis=1))
y = df["rating"]

onehot_columns = ct.named_transformers_['categorical'].get_feature_names(input_features=categorical_columns)

total_columns = numerical_columns + list(onehot_columns)
interesting_features_names = [name for name in total_columns if "item" not in name]
#
# random_grid = {#'bootstrap': [True, False],
#                'max_depth': [10, 50, 100, None],
#                'max_features': ['auto', 'sqrt'],
#                'min_samples_leaf': [1, 2, 4],
#                'min_samples_split': [2, 5, 10],
#                'n_estimators': [10, 100, 200]}
#
# parameters = {
#     "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
#     "min_samples_split": np.linspace(0.1, 0.5, 12),
#     "min_samples_leaf": np.linspace(0.1, 0.5, 12),
#     "max_depth":[3, 5, 8,20],
#     "max_features":["log2","sqrt"],
#     #"criterion": ["friedman_mse"],
#     "subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
#     "n_estimators":[10, 100, 200]
#     }
#
# rf_random = RandomizedSearchCV(estimator = clf,
#                                param_distributions = random_grid,
#                                scoring="neg_root_mean_squared_error",
#                                n_iter = 20,
#                                cv = 2,
#                                verbose=2,
#                                n_jobs = -1)
# # Fit the random search model
# rf_random.fit(X, y)
#
# print(rf_random.best_score_)
# print(rf_random.best_params_)

best_clf = RandomForestRegressor(
    **{'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 2, 'max_features': 'sqrt',
       'max_depth': 100})  # best param #TODO automate

#print(cross_validate(clf, X, y, cv=3, scoring=["neg_root_mean_squared_error", "neg_mean_absolute_error"]))
#print(cross_validate(best_clf, X, y, cv=3, scoring=["neg_root_mean_squared_error", "neg_mean_absolute_error"]))


import time
import numpy as np

best_clf.fit(X, y)
start_time = time.time()
importances = best_clf.feature_importances_
std = np.std([
    tree.feature_importances_ for tree in best_clf.estimators_], axis=0)
elapsed_time = time.time() - start_time

print(f"Elapsed time to compute the importances: "
      f"{elapsed_time:.3f} seconds")

forest_importances = pd.Series(importances[:len(interesting_features_names)], index = interesting_features_names)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std[:len(interesting_features_names)], ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
plt.show()
#TODO group by user and remove the mean
#+add the mean

#{'fit_time': array([11.08269691, 11.44243193, 15.29740596]), 'score_time': array([0.00889015, 0.00890326, 0.02219486]), 'test_neg_root_mean_squared_error': array([-2.53969294, -2.69247966, -2.87726253]), 'test_neg_mean_absolute_error': array([-2.10127254, -2.15607996, -2.40551145])}
from sklearn.inspection import permutation_importance
X_train, X_test, y_train, y_test = train_test_split(X, y)
start_time = time.time()
result = permutation_importance(
    best_clf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1, scoring="neg_root_mean_squared_error")
elapsed_time = time.time() - start_time
print(f"Elapsed time to compute the importances: "
      f"{elapsed_time:.3f} seconds")

forest_importances = pd.Series(result.importances_mean[:len(interesting_features_names)], index=interesting_features_names)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=result.importances_std[:len(interesting_features_names)], ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.show()