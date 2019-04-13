from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def save_to_csv(name, dataset, sep=','):
    dataset.to_csv(name + ".csv", sep=sep, encoding='utf-8')

def data_preperation(name, df):
    print('to complete')

def identify_and_set_correct_types(elections_df):
    features = elections_df.keys().drop('Vote')
    for feature in features:
        if elections_df[feature].dtype == 'object':
            elections_df[feature] = elections_df[feature].astype('category')


def main():
    # Data loading
    elections_df = pd.read_csv('ElectionsData.csv')
    # df.dtypes to watch the columns types

    # Split the data - 60% train, 20% validation, 20% test
    train_raw, validate_raw, test_raw = np.split(elections_df.sample(frac=1), [int(.6 * len(elections_df)),
                                                                     int(.8 * len(elections_df))])
    identify_and_set_correct_types(elections_df)

    train, validate, test = np.split(elections_df.sample(frac=1), [int(.6 * len(elections_df)),
                                                                   int(.8 * len(elections_df))])
    for dataset in [['train_raw', train_raw], ['validate_raw', validate_raw], ['test_raw', test_raw]]:
        save_to_csv(name=dataset[0], dataset=dataset[1])
    for dataset in [['train', train], ['validate', validate], ['test', test]]:
        data_preperation(name=dataset[0], df=dataset[1])

if __name__ == "__main__":
    main()
