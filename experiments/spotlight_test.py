from spotlight.cross_validation import random_train_test_split
from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.evaluation import rmse_score, mrr_score
from spotlight.factorization.explicit import ExplicitFactorizationModel

dataset = get_movielens_dataset(variant='100K')

train, test = random_train_test_split(dataset)

model = ExplicitFactorizationModel(n_iter=5)
model.fit(train)

rmse = rmse_score(model, test)

mrr = mrr_score(model, test)

print(rmse)

print(mrr)
