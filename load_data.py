#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 10:07:58 2022

@author: paul
"""

from sklearn.datasets import load_iris, load_digits, load_wine, load_breast_cancer
from sklearn.preprocessing import OneHotEncoder
from NN_class import NN


# iris_data = load_iris() 
# x = iris_data.data
# y_ = iris_data.target.reshape(-1, 1) 


# digits_data = load_digits(n_class=2) 
# x = digits_data.data
# y_ = digits_data.target.reshape(-1, 1)


# wine_data = load_wine() 
# x = wine_data.data
# y_ = wine_data.target.reshape(-1, 1) 


cancer_data = load_breast_cancer() 
x = cancer_data.data
y_ = cancer_data.target.reshape(-1, 1) 


# One Hot encode the class labels
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y_)


Neural = NN(.1,1000)
acc = Neural.train_test(x, y)
print("Score on test set from my NN:", acc)



from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
model = MLPClassifier(hidden_layer_sizes=(1), activation='logistic', learning_rate_init=.1, max_iter=1000)
model.fit(X_train, y_train);
score = model.score(X_test, y_test)
print("Score on test set from MLP:", score)