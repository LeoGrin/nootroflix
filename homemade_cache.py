import pickle
from train_model import *
#trying to resolve the memory leak I have on heroku

mean_ratings = compute_mean_ratings()
with open('saved_objects/mean_ratings.pickle', 'wb') as f:
    pickle.dump(mean_ratings, f)

avalaible_nootropics, item_baselines_inner, similarity_matrix, raw_to_iid, k, min_k, rating_lower, rating_upper = train_model()
with open('saved_objects/model_and_all.pickle', 'wb') as f:
    pickle.dump((avalaible_nootropics, item_baselines_inner, similarity_matrix, raw_to_iid, k, min_k, rating_lower, rating_upper), f)

