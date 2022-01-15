import pandas as pd
from spotlight.cross_validation import random_train_test_split
from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.evaluation import rmse_score, mrr_score
from spotlight.factorization.explicit import ExplicitFactorizationModel
from spotlight.interactions import Interactions
from sklearn.preprocessing import LabelEncoder
#dataset = get_movielens_dataset(variant='100K')

df = pd.read_csv("../data/total_df.csv")

le_item = LabelEncoder()
le_users = LabelEncoder()

dataset = Interactions(le_users.fit_transform(df["userID"].to_numpy()), le_item.fit_transform(df["itemID"].to_numpy()), ratings=df["rating"].to_numpy())


for _ in range(10):
    train, test = random_train_test_split(dataset)


    model = ExplicitFactorizationModel(n_iter=10, l2=0.001)
    model.fit(train)

    rmse = rmse_score(model, test)

    mrr = mrr_score(model, test)

    print(rmse)

    print(mrr)
