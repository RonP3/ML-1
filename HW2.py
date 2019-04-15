from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn import preprocessing


def save_to_csv(name, dataset, sep=','):
    dataset.to_csv(name + ".csv", sep=sep, encoding='utf-8')


def data_preperation(name, df):
    print('to complete')


def identify_and_set_correct_types(elections_df):
    le = preprocessing.LabelEncoder()
    features = elections_df.keys()

    for feature in features:
        if elections_df[feature].dtype == 'object':
            elections_df[feature] = le.fit_transform(elections_df[feature].astype(str))


def dummify_categories(elections_df):
    return pd.get_dummies(elections_df)


def impute(X, y):
    imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    transformed = imp.fit_transform(X, y)
    df_transformed = pd.DataFrame(transformed, columns=X.columns).astype(X.dtypes.to_dict())
    return df_transformed


def handle_corrupt_data(elections_df):
    # -122707.360023781 , -108232.195771866 , 160017.623874907 on AVG_lottary_expanses
    # -694.383176544183 on Avg_monthly_expense_when_under_age_21

    # elections_errs = elections_df[(elections_df.AVG_lottary_expanses == -122707.360023781) |
    #                               (elections_df.Avg_monthly_expense_when_under_age_21 == -694.383176544183)]
    elections_df.loc[
        elections_df['AVG_lottary_expanses'] == -122707.360023781, ['AVG_lottary_expanses']] = 0
    elections_df.loc[
        elections_df['AVG_lottary_expanses'] == -108232.195771866, ['AVG_lottary_expanses']] = 0
    elections_df.loc[
        elections_df['AVG_lottary_expanses'] == -160017.623874907, ['AVG_lottary_expanses']] = 0
    elections_df.loc[
        elections_df['Avg_monthly_expense_when_under_age_21'] == -694.383176544183, [
            'Avg_monthly_expense_when_under_age_21']] = 0


def main():
    # Data loading
    elections_df = pd.read_csv('ElectionsData.csv')
    # df.dtypes to watch the columns types

    # Split the data - 60% train, 20% validation, 20% test
    train_df, validate_df, test_df = np.split(elections_df.sample(frac=1), [int(.6 * len(elections_df)),
                                                                            int(.8 * len(elections_df))])

    for dataset in [['train_raw', train_df], ['validate_raw', validate_df], ['test_raw', test_df]]:
        save_to_csv(name=dataset[0], dataset=dataset[1])

    train_y_df = train_df.filter(["Vote"])
    train_df = train_df.drop(columns=["Vote"])

    train_df = impute(train_df, train_y_df)

    print("imputed")
    train_df = dummify_categories(train_df)
    print("dummied")

    # for name, df in [['train', train], ['validate', validate], ['test', test]]:
    #     train = identify_and_set_correct_types(df)

    for dataset in [['train_manipulated', train_df]]:
        save_to_csv(name=dataset[0], dataset=dataset[1])

    # for name, df in [['train', train_df], ['validate', validate_df], ['test', test_df]]:
    #     data_preperation(name=name, df=df)


if __name__ == "__main__":
    main()
