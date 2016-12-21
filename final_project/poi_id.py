#!/usr/bin/python
import sys
import cPickle as pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'bonus', 'long_term_incentive', 'deferred_income', 'deferral_payments',
                 'loan_advances', 'other', 'expenses', 'director_fees', 'total_payments', 
                 'exercised_stock_options', 'restricted_stock', 'restricted_stock_deferred', 
                 'total_stock_value', 'to_messages', 'from_messages', 'from_this_person_to_poi', 
                 'from_poi_to_this_person', 'shared_receipt_with_poi', 'fraction_from_poi', 'fraction_to_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
# turn into pandas dataframe
df = pd.DataFrame.from_dict(data_dict, orient='index')
df.replace('NaN', np.nan, inplace = True)

# remove TOTAL outlier
df.drop('TOTAL', inplace = True)

### Task 3: Create new feature(s)
df['fraction_from_poi'] = df['from_poi_to_this_person'] / df['to_messages']
df['fraction_to_poi'] = df['from_this_person_to_poi'] / df['from_messages']

### Store to my_dataset for easy export below.
# featureFormat expects 'NaN' strings
filled_df = df.fillna(value='NaN')

# back into dict
data_dict = filled_df.to_dict(orient='index')
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
y, X = targetFeatureSplit(data)
X = np.array(X)
y = np.array(y)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', SelectKBest()),
        ('reducer', PCA(random_state=42)),
        ('classifier', GaussianNB())
    ])

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

### Full hyperparameter search 
# SCALER = [None, StandardScaler()]
# SELECTOR__K = [10, 13, 15, 18, 'all']
# REDUCER__N_COMPONENTS = [2, 4, 6, 8, 10]

### Tuned parameters
SCALER = [None]
SELECTOR__K = [15]
REDUCER__N_COMPONENTS = [6]

param_grid = {
    'scaler': SCALER,
    'selector__k': SELECTOR__K,
    'reducer__n_components': REDUCER__N_COMPONENTS
}

sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

gnb_grid = GridSearchCV(pipe, param_grid, scoring='f1', cv=sss)
gnb_grid.fit(X, y)

clf = gnb_grid.best_estimator_

# Example starting point. Try investigating other evaluation techniques!
cv_accuracy = []
cv_precision = []
cv_recall = []
cv_f1 = []
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    
    cv_accuracy.append(accuracy_score(y_test, pred))
    cv_precision.append(precision_score(y_test, pred))
    cv_recall.append(recall_score(y_test, pred))
    cv_f1.append(f1_score(y_test, pred))
    
print "Mean Accuracy: {}".format(np.mean(cv_accuracy))
print "Mean Precision: {}".format(np.mean(cv_precision))
print "Mean Recall: {}".format(np.mean(cv_recall))
print "Mean f1: {}".format(np.mean(cv_f1))

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)