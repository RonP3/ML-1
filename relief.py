import numpy as np


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


def nearhitmiss(x_index, dists, y_df):
    miss_index = -1
    miss_dist = np.inf
    hit_index = -1
    hit_dist = np.inf
    for i in range(len(dists)):
        if y_df.iloc[i].item() == y_df.iloc[x_index].item() and dists[i] < hit_dist:
            hit_index = i
            hit_dist = dists[i]
        elif y_df.iloc[i].item() != y_df.iloc[x_index].item() and dists[i] < miss_dist:
            miss_index = i
            miss_dist = dists[i]
    return hit_index, miss_index


# calculates distance from input_df[x_index] to every other
# instance in the input_df
def calculate_distance(x_index, input_df):
    # dist = np.zeros(input_df.shape[0])
    dist = np.linalg.norm(input_df.iloc[x_index] - input_df, axis=1)
    # for i in range(input_df.shape[0]):
    #     dist[i] = np.linalg.norm(input_df.iloc[x_index] - input_df.iloc[i])
    return dist


# implementation of relief algorithm
# accepts 'value' for filtering values under threshold,
# and 'best' for returning #threshold best features
def relief(X_df, y_df, duration, threshold_type='value', threshold=0):
    w = np.zeros(X_df.shape[1])
    for t in range(duration):
        x_index = np.random.randint(0, len(X_df))
        x = X_df.iloc[x_index]
        dists = calculate_distance(x_index, X_df)
        nearhit_index, nearmiss_index = nearhitmiss(x_index, dists, y_df)
        w += np.power(x - X_df.iloc[nearmiss_index], 2) - \
             np.power(x - X_df.iloc[nearhit_index], 2)
    if threshold_type == 'value':
        return list(X_df.keys()[np.argwhere(w >= threshold)])
    elif threshold_type == 'best':
        return list(X_df.keys()[(-w).argsort()[:threshold]])

