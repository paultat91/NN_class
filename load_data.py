#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 10:07:58 2022

@author: paul
"""

from sklearn.datasets import load_iris, load_digits
from sklearn.preprocessing import OneHotEncoder
from NN_class import NN


# iris_data = load_iris() # load the iris dataset
# x = iris_data.data
# y_ = iris_data.target.reshape(-1, 1) # Convert data to a single column


digits_data = load_digits(n_class=2) # load the iris dataset
x = digits_data.data
y_ = digits_data.target.reshape(-1, 1) # Convert data to a single column


# One Hot encode the class labels
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y_)


Neural = NN(1,1000)
acc = Neural.train_test(x, y)
print(acc)