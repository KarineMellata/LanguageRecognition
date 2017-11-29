#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 14:37:32 2017

@author: mattchan
"""

import csv
import numpy as np
import random
import io
from pprint import pprint
from time import time

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MultiLabelBinarizer
#from sklearn.feature_selection import mutual_info_classif, chi2, SelectKBest
from sklearn.svm import LinearSVC
import sys 
import re

# enabling windows and macOS training with the w argument
if len(sys.argv)>1:
    if sys.argv[1] == 'w':
        train_set_x = 'comp_551_A2\train_set_x'
        train_set_y = 'comp_551_A2\train_set_y'
        test_set_x = 'comp_551_A2\test_set_x'
else:
    train_set_x = "./train_set_x.csv"
    train_set_y = "./train_set_y.csv"
    test_set_x = "./test_set_x.csv" 

y_train = []
raw_y = []
X_test = []

# load training data into memory
with io.open(train_set_x, "r", encoding='utf-8') as f:
    with io.open(train_set_y, 'r', encoding='utf-8') as f2:
    
        raw_y = list(csv.reader(f2, delimiter = ','))
        processed_y = np.array(raw_y[1:])
        y_train = processed_y[:,1]
        
        raw_data = list(csv.reader(f, delimiter = ','))
        processed_data = np.array(raw_data[1:])

# load test data into memory
with io.open(test_set_x, 'r', encoding='utf-8') as f3:
    raw_x_test = list(csv.reader(f3, delimiter = ','))
    processed_x_test = np.array(raw_x_test[1:])
    X_test = processed_x_test[:,1]

X_train = []
chars = []

# Shuffle the incoming strings and process out links
for string in processed_data[:,1]:
    
    if re.match('http', string):
        X_train += ['']
        continue
    else:
        chars = list(string)
        random.shuffle(chars)
        s = ''
        s = s.join(chars)
        X_train += [s]

#N_FEATURES_OPTIONS = [500, 1000, 2000, 5000]

#Pipeline implementation with feature selection - used Naive Bayes
text_clf = Pipeline(
            [('vect', CountVectorizer()),
             ('tfidf', TfidfTransformer(norm='l2')),
             ('clf', MultinomialNB(alpha=0.01)),
             ])
    
parameters = {
        'vect__ngram_range': [(1,3)],
        'vect__analyzer':['char'],
        # 'tfidf__use_idf':[True],
        'vect__max_features':[50000],
        # 'clf__alpha': (0.1, 0.001),
        # 'tfidf__smooth_idf': [True],
        # 'tfidf__sublinear_tf': [True],
#        'reduce_dim': [SelectKBest(mutual_info_classif)],
#        'reduce_dim__k': N_FEATURES_OPTIONS,
        'vect__max_df': [0.35],
        # 'clf__alpha':(0.1, 0.05, 0.01),
        # 'vect__ngram_range':[(1,1),(1,2),(1,3),(2,3)],
        'tfidf__use_idf':[True],
        # 'vect__max_features': (10000, 50000, 80000),
        }

if __name__ == "__main__":
    # Perform Gridsearch with the parameters and allow multiprocessing (need to turn this off for Windows)
    grid_search = GridSearchCV(text_clf, parameters, n_jobs=-1, verbose=1, cv=5)
    
    print("Performing grid search...")
    print("pipeline:", [name for name, _ in text_clf.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    # fit and time the gridsearch
    grid_search.fit(X_train, y_train)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

# use the best trained model to predict on the test set
results = grid_search.predict(X_test)

# Write the predictions to a csv file
with io.open('predictions.csv', 'w', encoding='utf-8') as predictions:
    predictions.write('Id,Category\n')
    trial_num = 0
    for category in results:
        predictions.write(str(trial_num))
        predictions.write(',')
        predictions.write(str(results[trial_num]))
        predictions.write('\n')
        trial_num+=1

sys.stdout.write('\a')
sys.stdout.flush()
