import numpy as np
from HW2 import calculate_distance


def nearmiss(x_index, dists, y_df):
    miss_index = -1
    miss_dist = np.inf
    for i in range(len(dists)):
        if y_df.iloc[i].item() != y_df.iloc[x_index].item() and dists[i] < miss_dist:
            miss_index = i
            miss_dist = dists[i]
    return miss_index


def nearhit(x_index, dists, y_df):
    hit_index = -1
    hit_dist = np.inf
    for i in range(len(dists)):
        if y_df.iloc[i].item() == y_df.iloc[x_index].item() and dists[i] < hit_dist:
            hit_index = i
            hit_dist = dists[i]
    return hit_index


def relief(X_df, y_df):
    w = np.zeros(X_df.shape[1])
    for t in range(len(X_df.index)):
        x_index = np.random.randint(0, len(X_df))
        x = X_df.iloc[x_index]
        dists = calculate_distance(x_index, X_df)
        nearmiss_index = nearmiss(x_index, dists, y_df)
        nearhit_index = nearhit(x_index, dists, y_df)
        for i in range(X_df.shape[1]):
            w[i] += np.power(x[i] - X_df.iloc[nearmiss_index][i], 2) - \
                    np.power(x[i] - X_df.iloc[nearhit_index][i], 2)
        print("t", t)
        print("w", w)
    print("final w", w)
