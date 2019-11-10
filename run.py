import os

from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import pandas
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

from sklearn import preprocessing

train_url = os.path.abspath("data/TrainingData.csv")
names = ['id', 'age', 'CurrentResidenceYears', 'isMarried', 'NumberOfDependants', 'Graduated', 'SelfEmployed',
         'YearsOfJobStability', 'YearlySalary', 'CreditRating', 'CoApplicantAge', 'CoApplicantYearsOfJobStability',
         'CoApplicantYearlySalary', 'CoApplicantCreditRating', 'LoanTermInYears', 'LoanAmount', 'PropertyTotalCost', 'AreaClassification', 'Approved']
raw_dataset = read_csv(train_url, names=names, nrows=200000)
del raw_dataset['id']

codepedent_dataset = raw_dataset[(raw_dataset["CoApplicantAge"] > 0)]
codepedent_dataset['age'] = codepedent_dataset['age'].astype(float)
codepedent_dataset['CurrentResidenceYears'] = codepedent_dataset['CurrentResidenceYears'].astype(float)
codepedent_dataset['NumberOfDependants'] = codepedent_dataset['NumberOfDependants'].astype(float)
codepedent_dataset['YearsOfJobStability'] = codepedent_dataset['YearsOfJobStability'].astype(float)
codepedent_dataset['YearlySalary'] = codepedent_dataset['YearlySalary'].astype(float)
codepedent_dataset['CoApplicantAge'] = codepedent_dataset['CoApplicantAge'].astype(float)
codepedent_dataset['CoApplicantYearsOfJobStability'] = codepedent_dataset['CoApplicantYearsOfJobStability'].astype(float)
codepedent_dataset['CoApplicantYearlySalary'] = codepedent_dataset['CoApplicantYearlySalary'].astype(float)
codepedent_dataset['LoanTermInYears'] = codepedent_dataset['LoanTermInYears'].astype(float)
codepedent_dataset['LoanAmount'] = codepedent_dataset['LoanAmount'].astype(float)
codepedent_dataset['PropertyTotalCost'] = codepedent_dataset['PropertyTotalCost'].astype(float)

codepedent_dataset['isMarried'] = codepedent_dataset['isMarried'].astype(int).astype(float)
codepedent_dataset['Graduated'] = codepedent_dataset['Graduated'].astype(int).astype(float)
codepedent_dataset['SelfEmployed'] = codepedent_dataset['SelfEmployed'].astype(int).astype(float)
codepedent_dataset['Approved'] = codepedent_dataset['Approved'].astype(int).astype(float)


credit_rating_dict = {"AAA": 1, "AA": 2, "A": 3, "B": 4, "C": 5, "D": 6}
codepedent_dataset['CreditRating'] = codepedent_dataset['CreditRating'].replace(credit_rating_dict, inplace=False).astype(float)
codepedent_dataset['CoApplicantCreditRating'] = codepedent_dataset['CoApplicantCreditRating'].replace(credit_rating_dict, inplace=False).astype(float)

area_classification_dict = {"CLASS_A": 1, "CLASS_B": 2, "CLASS_C": 3}
codepedent_dataset['AreaClassification'] = codepedent_dataset['AreaClassification'].replace(area_classification_dict, inplace=False).astype(float)

normalized_codepedent_dataset = preprocessing.normalize(codepedent_dataset)
scaled_codepedent_dataset = preprocessing.scale(codepedent_dataset)


non_codependent_dataset = raw_dataset[(raw_dataset["CoApplicantAge"] == 0)]
del non_codependent_dataset['CoApplicantAge']
del non_codependent_dataset['CoApplicantYearsOfJobStability']
del non_codependent_dataset['CoApplicantYearlySalary']
del non_codependent_dataset['CoApplicantCreditRating']

non_codependent_dataset['age'] = non_codependent_dataset['age'].astype(float)
non_codependent_dataset['CurrentResidenceYears'] = non_codependent_dataset['CurrentResidenceYears'].astype(float)
non_codependent_dataset['NumberOfDependants'] = non_codependent_dataset['NumberOfDependants'].astype(float)
non_codependent_dataset['YearsOfJobStability'] = non_codependent_dataset['YearsOfJobStability'].astype(float)
non_codependent_dataset['YearlySalary'] = non_codependent_dataset['YearlySalary'].astype(float)
non_codependent_dataset['LoanTermInYears'] = non_codependent_dataset['LoanTermInYears'].astype(float)
non_codependent_dataset['LoanAmount'] = non_codependent_dataset['LoanAmount'].astype(float)
non_codependent_dataset['PropertyTotalCost'] = non_codependent_dataset['PropertyTotalCost'].astype(float)

non_codependent_dataset['isMarried'] = non_codependent_dataset['isMarried'].astype(int).astype(float)
non_codependent_dataset['Graduated'] = non_codependent_dataset['Graduated'].astype(int).astype(float)
non_codependent_dataset['SelfEmployed'] = non_codependent_dataset['SelfEmployed'].astype(int).astype(float)
non_codependent_dataset['Approved'] = non_codependent_dataset['Approved'].astype(int).astype(float)

non_codependent_dataset['CreditRating'] = non_codependent_dataset['CreditRating'].replace(credit_rating_dict, inplace=False).astype(float)

non_codependent_dataset['AreaClassification'] = non_codependent_dataset['AreaClassification'].replace(area_classification_dict, inplace=False).astype(float)

normalized_non_codependent_dataset = preprocessing.normalize(non_codependent_dataset)
scaled_non_codependent_dataset = preprocessing.scale(non_codependent_dataset)


# WE HAVE DATAAAAAAA

print("***********CODEPENDENT DATA***********")

array_codependent = codepedent_dataset.values
X_codependent = array_codependent[:, 0:17]
y_codependent = array_codependent[:, 17]

X_codependent_train, X_codependent_validation, Y_codependent_train, Y_codependent_validation = train_test_split(X_codependent, y_codependent, test_size=0.20, random_state=1)

models = []
models.append(('KM', KMeans()))
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
#models.append(('SVM', SVC(gamma='auto')))

results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1)
    cv_results = cross_val_score(model, X_codependent_train, Y_codependent_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))


print("***********CODEPENDENT NORMALIZED DATA***********")

array_codependent_normalized = normalized_codepedent_dataset
X_codependent_normalized = array_codependent[:, 0:17]
y_codependent_normalized = array_codependent[:, 17]

X_codependent_normalized_train, X_codependent_normalized_validation, Y_codependent_normalized_train, Y_codependent_normalized_validation = train_test_split(X_codependent_normalized, y_codependent_normalized, test_size=0.20, random_state=1)

models = []
models.append(('KM', KMeans()))
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
#models.append(('SVM', SVC(gamma='auto')))

results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1)
    cv_results = cross_val_score(model, X_codependent_train, Y_codependent_normalized_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
