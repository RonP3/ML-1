from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn import preprocessing, tree
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.naive_bayes import GaussianNB
from sequential_selection import sfs, sbs, bds
from relief import relief
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


class DataPreparator:

    def __init__(self, df, target):
        self.df = df
        self.target = target
        self.train_df, self.validate_df, self.test_df = np.split(self.df.sample(frac=1), [int(.6 * len(self.df)),
                                                                                          int(.8 * len(self.df))])

        self.train_y_df = self.train_df.filter([self.target])
        self.validate_y_df = self.validate_df.filter([self.target])
        self.test_y_df = self.test_df.filter([self.target])

    def save_to_file(self, suffix=''):
        for dataset in [['train_' + suffix, self.train_df], ['validate_' + suffix, self.validate_df],
                        ['test_' + suffix, self.test_df]]:
            save_to_csv(name=dataset[0], dataset=dataset[1])

    def process_data_for_df(self, X):
        y = X.filter([self.target])
        X = X.drop(columns=[self.target])
        X = self.handle_negative_data(X)
        X = self.impute(X, y)
        X = self.dummify_categories(X)
        X = self.scale(X)
        return X, y

    def update_selected_features(self, selected):
        selected = set(selected).intersection(set(self.train_df.keys())).intersection(
            set(self.validate_df.keys()).intersection(set(self.test_df.keys())))

        self.train_df = self.train_df[selected]
        self.validate_df = self.validate_df[selected]
        self.test_df = self.test_df[selected]

    def process_data(self):
        t1, t2 = self.process_data_for_df(self.train_df)
        v1, v2 = self.process_data_for_df(self.validate_df)
        r1, r2 = self.process_data_for_df(self.test_df)

        self.train_df, self.train_y_df = t1 , t2
        self.validate_df, self.validate_y_df = v1, v2
        self.test_df, self.test_y_df = r1, r2

    def identify_and_set_correct_types(self):
        features = set(self.train_df.keys()) - {self.target}

        numerical, categorical = [], []
        for feature in features:
            if self.train_df[feature].dtype == 'float' or self.train_df[feature].dtype == 'int':
                numerical.append(feature)
            else:
                categorical.append(feature)
        return numerical, categorical

    def impute(self, X, y):
        numerical, _ = self.identify_and_set_correct_types()
        numerical = set(numerical).intersection(set(X.keys()))
        categorical = set(X.keys()) - set(numerical)
        X_categorical = X[categorical]
        imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        #  calculating imputation values with train df only
        imp.fit(self.train_df[categorical], y)
        transformed = imp.transform(X_categorical)
        categorical_transformed = pd.DataFrame(transformed, columns=X_categorical.columns).astype(
            X_categorical.dtypes.to_dict())

        X_numerical = X[numerical]
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        # calculating imputation values with train df only
        imp.fit(self.train_df[numerical], y)
        transformed = imp.transform(X_numerical)
        numerical_transformed = pd.DataFrame(transformed, columns=X_numerical.columns).astype(
            X_numerical.dtypes.to_dict())

        return pd.concat([categorical_transformed, numerical_transformed], axis=1)

    def dummify_categories(self, X):
        return pd.get_dummies(X)

    def handle_corrupt_data(self):
        self.df.loc[
            self.df < 0] = np.nan

    def scale(self, df):
        # uniform dist scale
        numerical, categorical = self.identify_and_set_correct_types()
        numerical = set(numerical).intersection(set(df.keys()))

        uniform_features = df[list(set(uniform_dist_features).intersection(numerical))]
        df[uniform_features.keys()] = (preprocessing.MinMaxScaler(feature_range=(-1, 1),
                                                                  copy=False).fit_transform(uniform_features))

        # normal dist scale
        normal_features = df[
            list(set(normal_dist_features).union(set(other_dist_features)).intersection(numerical))]
        df[normal_features.keys()] = (df[normal_features.keys()] -
                                      df[normal_features.keys()].mean()) / df[normal_features.keys()].std()
        return df

    def handle_negative_data(self, X):
        numeric_columns = X.select_dtypes([np.float64]).columns.data.obj

        for column in numeric_columns:
            X.loc[X[column] < 0, column] = np.nan
        return X

    def relief(self, save=False, load=False, test=False, iterates=5, threshold_type='best', threshold=30):
        if load:
            output = open('relief.pkl', 'rb')
            selected = pickle.load(output)
            output.close()
            return selected

        selected = relief(self.train_df, self.train_y_df, iterates, threshold_type, threshold)
        if test:
            for classifier in {tree.DecisionTreeClassifier, GaussianNB, SVC, KNN}:
                clf = classifier()
                clf.fit(self.train_df, np.ravel(self.train_y_df))
                print(str(classifier), " before Relief: ", clf.score(self.validate_df, self.validate_y_df))
                clf.fit(self.train_df[selected], np.ravel(self.train_y_df))
                print(str(classifier), " after Relief:", len(selected), " ",
                      clf.score(self.validate_df[selected], self.validate_y_df))

        if save:
            output = open('relief.pkl', 'wb')
            pickle.dump(selected, output)
            output.close()

        return selected

    def sfs(self, save=False, load=False, test=False, classifier=SVC(gamma='auto')):

        if load:
            output = open('sfs.pkl', 'rb')
            selected = pickle.load(output)
            output.close()
            return selected

        selected = sfs(self.train_df, np.ravel(self.train_y_df), self.validate_df, np.ravel(self.validate_y_df),
                       classifier)
        if test:
            for classifier in {tree.DecisionTreeClassifier, GaussianNB, SVC, KNN}:
                clf = classifier()
                clf.fit(self.train_df, np.ravel(self.train_y_df))
                print(str(classifier), " before sfs: ", clf.score(self.validate_df, self.validate_y_df))
                clf.fit(self.train_df[selected], np.ravel(self.train_y_df))
                print(str(classifier), " after sfs:", len(selected), " ",
                      clf.score(self.validate_df[selected], self.validate_y_df))

        if save:
            output = open('sfs.pkl', 'wb')
            pickle.dump(selected, output)
            output.close()

        return selected


def save_to_csv(name, dataset, sep=','):
    dataset.to_csv(name + ".csv", sep=sep, encoding='utf-8')


def main():
    elections_df = pd.read_csv('ElectionsData.csv')
    dp = DataPreparator(elections_df, 'Vote')
    print("saving raw data")
    dp.save_to_file('raw')
    dp.process_data()
    print("finished processing")
    print("starting relief")
    selected = dp.relief(save=True, load=False, test=True, iterates=1000, threshold_type='best', threshold=30)
    print("finished relief")
    dp.update_selected_features(selected)
    print("starting sfs")
    selected = dp.sfs(save=True, load=False, test=True, classifier=SVC())
    print("finished sfs")
    dp.update_selected_features(selected)
    dp.save_to_file('processed')


if __name__ == "__main__":
    main()
