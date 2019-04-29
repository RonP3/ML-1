import numpy as np
import pandas as pd

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
    numeric_columns = input_df.select_dtypes([np.float64]).columns.data.obj
    other_keys = set(input_df.keys()) - set(numeric_columns)

    num_dist = np.linalg.norm(input_df[numeric_columns].iloc[x_index] - input_df[numeric_columns], axis=1)
    disc_dist = 0
    if other_keys:
        disc_dist = np.linalg.norm(input_df[other_keys].ne(input_df[other_keys].iloc[x_index], axis=1), axis=1)
    return np.sqrt(np.square(num_dist) + np.square(disc_dist))


# implementation of relief algorithm
# accepts 'value' for filtering values under threshold,
# and 'best' for returning #threshold best features
def relief(X_df, y_df, duration, threshold_type='value', threshold=0, threshold_disc=0):
    numeric_columns = list(X_df.select_dtypes([np.float64]).columns.data.obj)
    other_keys_set = set(X_df.keys()) - set(numeric_columns)
    other_keys = list(other_keys_set)

    if duration != 0:
        w = pd.DataFrame(0, index=np.arange(1), columns=X_df.keys())
        for t in range(duration):
            x_index = np.random.randint(0, len(X_df))
            x = X_df.iloc[x_index]
            dists = calculate_distance(x_index, X_df)
            nearhit_index, nearmiss_index = nearhitmiss(x_index, dists, y_df)
            w[numeric_columns] += np.power(x[numeric_columns] - X_df[numeric_columns].iloc[nearmiss_index], 2) - \
                 np.power(x[numeric_columns] - X_df[numeric_columns].iloc[nearhit_index], 2)
            # categorical hamming distance -
            w[other_keys] += X_df[other_keys].iloc[nearmiss_index].ne(x[other_keys]).astype(float) - X_df[other_keys].iloc[
                nearhit_index].ne(x[other_keys]).astype(float)

    else:
        w = pd.read_csv("w.csv", sep=',', encoding='utf-8', header=0)
    if threshold_type == 'value':
        num_pass = set([numeric_columns[x] for x in np.argwhere(w[numeric_columns].iloc[0] >= threshold).flatten()])
        cat_pass = set([other_keys[x] for x in np.argwhere(w[other_keys].iloc[0] >= threshold_disc).flatten()])
        return num_pass.union(cat_pass)
    elif threshold_type == 'best':
        return set(X_df.keys()[(-w.iloc[0]).argsort()[:min(threshold, len(w.iloc[0]))]-1])
