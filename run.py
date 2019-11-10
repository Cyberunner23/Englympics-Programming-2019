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

coapplicant_dataset = raw_dataset[(raw_dataset["CoApplicantAge"] > 0)]
coapplicant_dataset['age'] = coapplicant_dataset['age'].astype(float)
coapplicant_dataset['CurrentResidenceYears'] = coapplicant_dataset['CurrentResidenceYears'].astype(float)
coapplicant_dataset['NumberOfDependants'] = coapplicant_dataset['NumberOfDependants'].astype(float)
coapplicant_dataset['YearsOfJobStability'] = coapplicant_dataset['YearsOfJobStability'].astype(float)
coapplicant_dataset['YearlySalary'] = coapplicant_dataset['YearlySalary'].astype(float)
coapplicant_dataset['CoApplicantAge'] = coapplicant_dataset['CoApplicantAge'].astype(float)
coapplicant_dataset['CoApplicantYearsOfJobStability'] = coapplicant_dataset['CoApplicantYearsOfJobStability'].astype(float)
coapplicant_dataset['CoApplicantYearlySalary'] = coapplicant_dataset['CoApplicantYearlySalary'].astype(float)
coapplicant_dataset['LoanTermInYears'] = coapplicant_dataset['LoanTermInYears'].astype(float)
coapplicant_dataset['LoanAmount'] = coapplicant_dataset['LoanAmount'].astype(float)
coapplicant_dataset['PropertyTotalCost'] = coapplicant_dataset['PropertyTotalCost'].astype(float)

coapplicant_dataset['isMarried'] = coapplicant_dataset['isMarried'].astype(int).astype(float)
coapplicant_dataset['Graduated'] = coapplicant_dataset['Graduated'].astype(int).astype(float)
coapplicant_dataset['SelfEmployed'] = coapplicant_dataset['SelfEmployed'].astype(int).astype(float)
coapplicant_dataset['Approved'] = coapplicant_dataset['Approved'].astype(int).astype(float)


credit_rating_dict = {"AAA": 1, "AA": 2, "A": 3, "B": 4, "C": 5, "D": 6}
coapplicant_dataset['CreditRating'] = coapplicant_dataset['CreditRating'].replace(credit_rating_dict, inplace=False).astype(float)
coapplicant_dataset['CoApplicantCreditRating'] = coapplicant_dataset['CoApplicantCreditRating'].replace(credit_rating_dict, inplace=False).astype(float)

area_classification_dict = {"CLASS_A": 1, "CLASS_B": 2, "CLASS_C": 3}
coapplicant_dataset['AreaClassification'] = coapplicant_dataset['AreaClassification'].replace(area_classification_dict, inplace=False).astype(float)


non_coapplicant_dataset = raw_dataset[(raw_dataset["CoApplicantAge"] == 0)]
del non_coapplicant_dataset['CoApplicantAge']
del non_coapplicant_dataset['CoApplicantYearsOfJobStability']
del non_coapplicant_dataset['CoApplicantYearlySalary']
del non_coapplicant_dataset['CoApplicantCreditRating']

non_coapplicant_dataset['age'] = non_coapplicant_dataset['age'].astype(float)
non_coapplicant_dataset['CurrentResidenceYears'] = non_coapplicant_dataset['CurrentResidenceYears'].astype(float)
non_coapplicant_dataset['NumberOfDependants'] = non_coapplicant_dataset['NumberOfDependants'].astype(float)
non_coapplicant_dataset['YearsOfJobStability'] = non_coapplicant_dataset['YearsOfJobStability'].astype(float)
non_coapplicant_dataset['YearlySalary'] = non_coapplicant_dataset['YearlySalary'].astype(float)
non_coapplicant_dataset['LoanTermInYears'] = non_coapplicant_dataset['LoanTermInYears'].astype(float)
non_coapplicant_dataset['LoanAmount'] = non_coapplicant_dataset['LoanAmount'].astype(float)
non_coapplicant_dataset['PropertyTotalCost'] = non_coapplicant_dataset['PropertyTotalCost'].astype(float)

non_coapplicant_dataset['isMarried'] = non_coapplicant_dataset['isMarried'].astype(int).astype(float)
non_coapplicant_dataset['Graduated'] = non_coapplicant_dataset['Graduated'].astype(int).astype(float)
non_coapplicant_dataset['SelfEmployed'] = non_coapplicant_dataset['SelfEmployed'].astype(int).astype(float)
non_coapplicant_dataset['Approved'] = non_coapplicant_dataset['Approved'].astype(int).astype(float)

non_coapplicant_dataset['CreditRating'] = non_coapplicant_dataset['CreditRating'].replace(credit_rating_dict, inplace=False).astype(float)

non_coapplicant_dataset['AreaClassification'] = non_coapplicant_dataset['AreaClassification'].replace(area_classification_dict, inplace=False).astype(float)


# WE HAVE DATAAAAAAA

# Used to test which model is the best, DecisionTreeClassifier wins
print("***********COAPPLICANT DATA***********")

array_coapplicant = coapplicant_dataset.values
X_coapplicant = array_coapplicant[:, 0:17]
y_coapplicant = array_coapplicant[:, 17]

X_coapplicant_train, X_coapplicant_validation, Y_coapplicant_train, Y_coapplicant_validation = train_test_split(X_coapplicant, y_coapplicant, test_size=0.20, random_state=1)

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
    cv_results = cross_val_score(model, X_coapplicant_train, Y_coapplicant_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))


print("***********NON COAPPLICANT NORMALIZED DATA***********")

array_non_coapplicant = non_coapplicant_dataset.values
X_non_coapplicant = array_non_coapplicant[:, 0:13]
y_non_coapplicant = array_non_coapplicant[:, 13]

X_non_coapplicant_train, X_non_coapplicant_validation, Y_non_coapplicant_train, Y_non_coapplicant_validation = train_test_split(X_non_coapplicant, y_non_coapplicant, test_size=0.20, random_state=1)

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
    cv_results = cross_val_score(model, X_non_coapplicant_train, Y_non_coapplicant_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))


# TRAINING ACTUAL MODEL

# coapplicant
array_coapplicant = coapplicant_dataset.values
X_coapplicant = array_coapplicant[:, 0:17]
y_coapplicant = array_coapplicant[:, 17]

model_coapplicant = DecisionTreeClassifier()
model_coapplicant.fit(X_coapplicant, y_coapplicant)

# noncoapplicant
array_non_coapplicant = non_coapplicant_dataset.values
X_non_coapplicant = array_non_coapplicant[:, 0:13]
y_non_coapplicant = array_non_coapplicant[:, 13]

model_non_coapplicant = DecisionTreeClassifier()
model_non_coapplicant.fit(X_non_coapplicant, y_non_coapplicant)

# load prediction data
eval_url = os.path.abspath("data/ForEvaluation.csv")
names = ['id', 'age', 'CurrentResidenceYears', 'isMarried', 'NumberOfDependants', 'Graduated', 'SelfEmployed',
         'YearsOfJobStability', 'YearlySalary', 'CreditRating', 'CoApplicantAge', 'CoApplicantYearsOfJobStability',
         'CoApplicantYearlySalary', 'CoApplicantCreditRating', 'LoanTermInYears', 'LoanAmount', 'PropertyTotalCost', 'AreaClassification']
eval_dataset = read_csv(eval_url, names=names)

eval_dataset['age'] = eval_dataset['age'].astype(float)
eval_dataset['CurrentResidenceYears'] = eval_dataset['CurrentResidenceYears'].astype(float)
eval_dataset['NumberOfDependants'] = eval_dataset['NumberOfDependants'].astype(float)
eval_dataset['YearsOfJobStability'] = eval_dataset['YearsOfJobStability'].astype(float)
eval_dataset['YearlySalary'] = eval_dataset['YearlySalary'].astype(float)
eval_dataset['LoanTermInYears'] = eval_dataset['LoanTermInYears'].astype(float)
eval_dataset['LoanAmount'] = eval_dataset['LoanAmount'].astype(float)
eval_dataset['PropertyTotalCost'] = eval_dataset['PropertyTotalCost'].astype(float)
eval_dataset['isMarried'] = eval_dataset['isMarried'].astype(int).astype(float)
eval_dataset['Graduated'] = eval_dataset['Graduated'].astype(int).astype(float)
eval_dataset['SelfEmployed'] = eval_dataset['SelfEmployed'].astype(int).astype(float)
eval_dataset['CreditRating'] = eval_dataset['CreditRating'].replace(credit_rating_dict, inplace=False).astype(float)
eval_dataset['AreaClassification'] = eval_dataset['AreaClassification'].replace(area_classification_dict, inplace=False).astype(float)
eval_dataset['CoApplicantAge'] = eval_dataset['CoApplicantAge'].astype(float)
eval_dataset['CoApplicantYearsOfJobStability'] = eval_dataset['CoApplicantYearsOfJobStability'].astype(float)
eval_dataset['CoApplicantYearlySalary'] = eval_dataset['CoApplicantYearlySalary'].astype(float)
eval_dataset['CoApplicantCreditRating'] = eval_dataset['CoApplicantCreditRating'].replace(credit_rating_dict, inplace=False).astype(float)


with open("results.csv", "w") as f:
    for index, row in eval_dataset.iterrows():
        if row["CoApplicantAge"] == 0: # non coapplicant
            id = row['id']
            del row['id']
            del row['CoApplicantAge']
            del row['CoApplicantYearsOfJobStability']
            del row['CoApplicantYearlySalary']
            del row['CoApplicantCreditRating']
            f.write('{},{}\n'.format(id, bool(model_non_coapplicant.predict([row])[0])))
        else: # coapplicant
            id = row['id']
            del row['id']
            f.write('{},{}\n'.format(id, bool(model_coapplicant.predict([row])[0])))
