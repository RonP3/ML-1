from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn import preprocessing, tree
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.naive_bayes import GaussianNB
from sequential_selection import sfs, sbs, bds
import pickle


# feature distribution
normal_dist_features = ['Political_interest_Total_Score', 'Number_of_valued_Kneset_members',
                        'Number_of_differnt_parties_voted_for', 'Avg_size_per_room',
                        'Avg_Satisfaction_with_previous_vote', 'Avg_monthly_household_cost',
                        'Avg_monthly_expense_on_pets_or_plants', 'Avg_government_satisfaction']

uniform_dist_features = ['Yearly_IncomeK', 'Yearly_ExpensesK', 'Overall_happiness_score', 'Occupation',
                         'Occupation_Satisfaction', 'Main_transportation', 'Looking_at_poles_results', 'Gender',
                         'Garden_sqr_meter_per_person_in_residancy_area', 'Financial_balance_score_(0-1)',
                         'Financial_agenda_matters', 'Age_group', '%Time_invested_in_work', '%Of_Household_Income',
                         '%_satisfaction_financial_policy']

other_dist_features = ['Will_vote_only_large_party', 'Voting_Time', 'Phone_minutes_10_years',
                       'Num_of_kids_born_last_10_years', 'Most_Important_Issue', 'Married', 'Last_school_grades',
                       'Avg_Residancy_Altitude', 'Avg_monthly_income_all_years',
                       'Avg_monthly_expense_when_under_age_21', 'AVG_lottary_expanses', 'Avg_environmental_importance',
                       'Avg_education_importance', 'Weighted_education_rank']

categorical_features = ['Most_Important_Issue', 'Looking_at_poles_results', 'Married', 'Gender', 'Voting_Time',
                        'Will_vote_only_large_party', 'Age_group', 'Main_transportation', 'Occupation',
                        'Financial_agenda_matters']

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
        elections_df['AVG_lottary_expanses'] < 0, ['AVG_lottary_expanses']] = np.nan
    elections_df.loc[
        elections_df['Avg_monthly_expense_when_under_age_21'] < 0, [
            'Avg_monthly_expense_when_under_age_21']] = np.nan


def scale(df):
    # uniform dist scale
    uniform_features = df[list(set(uniform_dist_features)-set(categorical_features))]
    df[uniform_features.keys()] = (preprocessing.MinMaxScaler(feature_range=(-1, 1),
                                                              copy=False).fit_transform(uniform_features))

    # normal dist scale
    normal_features = df[list(set(normal_dist_features)-set(categorical_features))]
    df[normal_features.keys()] = (df[normal_features.keys()] -
                                  df[normal_features.keys()].mean())/df[normal_features.keys()].std()
    df[normal_features.keys()] = (preprocessing.MinMaxScaler(feature_range=(-1, 1),
                                                              copy=False).fit_transform(normal_features))
    # other dist scale
    other_features = df[list(set(other_dist_features) - set(categorical_features))]
    df[other_features.keys()] = (preprocessing.MinMaxScaler(feature_range=(-1, 1),
                                                              copy=False).fit_transform(other_features))
    return df


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

    # train_df = scale(train_df)
    # print("scaled")

    #  manipulate validation set
    validate_y_df = validate_df.filter(["Vote"])
    validate_df = validate_df.drop(columns=["Vote"])

    validate_df = impute(validate_df, validate_y_df)
    print("imputed")

    validate_df = dummify_categories(validate_df)
    print("dummied")

    # train_df = scale(train_df)
    # print("scaled")

    selected = bds(train_df, np.ravel(train_y_df), validate_df, np.ravel(validate_y_df), tree.DecisionTreeClassifier())
    print("selected: ", selected)

    for classifier in {tree.DecisionTreeClassifier, GaussianNB, SVC, KNN}:
        clf = classifier()
        clf.fit(train_df, np.ravel(train_y_df))
        print(str(classifier), " before :", clf.score(validate_df, validate_y_df))
        clf.fit(train_df[selected], np.ravel(train_y_df))
        print(str(classifier), " after :", len(selected), " ", clf.score(validate_df[selected], validate_y_df))

    # relief(train_df, train_y_df)

    # for name, df in [['train', train], ['validate', validate], ['test', test]]:
    #     train = identify_and_set_correct_types(df)
    #
    # for dataset in [['train_manipulated', train_df]]:
    #     save_to_csv(name=dataset[0], dataset=dataset[1])
    #
    # for name, df in [['train', train_df], ['validate', validate_df], ['test', test_df]]:
    #     data_preperation(name=name, df=df)


# calculates distance from input_df[x_index] to every other
# instance in the input_df
def calculate_distance(x_index, input_df):
    dist = np.zeros(input_df.shape[0])
    for i in range(input_df.shape[0]):
        dist[i] = np.linalg.norm(input_df.iloc[x_index] - input_df.iloc[i])
    return dist


if __name__ == "__main__":
    main()
