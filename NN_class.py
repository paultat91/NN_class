#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 21:23:02 2021

@author: paul
"""

from autograd import grad 
import autograd.numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


iris_data = load_iris() # load the iris dataset

x = iris_data.data
y_ = iris_data.target.reshape(-1, 1) # Convert data to a single column

# One Hot encode the class labels
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y_)


class NN(object):
    
    def __init__(self, learning_rate, lessons):
        self.learning_rate = learning_rate
        self.lessons = lessons        
        return
    
    def activation(self, x):
        return np.tanh(x)
    
    def softmax(self, x):
        exp = np.exp(x) 
        return exp / exp.sum(0)
    
    def Error(self, W_hh, W_oh, b_h, b_o, x, y):
        h = self.activation(np.dot(W_hh,x.T) + b_h)
        y_hat = self.softmax(np.dot(W_oh,h) + b_o)
        return np.sum(np.diag(np.dot(y, -np.log(y_hat))))/len(x)
    
    def forward(self, x, y, W_hh, W_oh, b_h, b_o):
        h = self.activation(np.dot(W_hh,x.T) + b_h)
        y_hat = self.softmax(np.dot(W_oh,h) + b_o)
        pred = np.expand_dims(np.argmax(y_hat, axis=0), axis=0).T
        num_wrong = np.count_nonzero(encoder.inverse_transform(y) - pred)
        acc = (len(x) - num_wrong)/len(x)
        err =self.Error(W_hh, W_oh, b_h, b_o, x, y) 
        return acc, err
    
    def update(self, W_hh, W_oh, b_h, b_o, x, y):
        dE_dWhh = grad(self.Error, argnum=0)(W_hh, W_oh, b_h, b_o, x, y)
        dE_dWoh = grad(self.Error, argnum=1)(W_hh, W_oh, b_h, b_o, x, y)    
        dE_dbh = grad(self.Error, argnum=2)(W_hh, W_oh, b_h, b_o, x, y)
        dE_dbo = grad(self.Error, argnum=3)(W_hh, W_oh, b_h, b_o, x, y)    

        W_hh = W_hh - self.learning_rate*dE_dWhh
        W_oh = W_oh - self.learning_rate*dE_dWoh  

        b_h = b_h - self.learning_rate*dE_dbh
        b_o = b_o - self.learning_rate*dE_dbo          
        return W_hh, W_oh, b_h, b_o
    
    def train_test(self, x, y):
        
        self.initial_weights = np.random.rand(1,7) #np.ones([1, 7]) 
        self.final_weights = []     
        
        # # Split the data for training and testing
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.20, random_state=56)
        
        #Standardization
        for i in range(len(train_x[0])):
            train_x[:,i] = (train_x[:,i] - np.mean(train_x[:,i]))/np.std(train_x[:,i])    
            test_x[:,i] = (test_x[:,i] - np.mean(test_x[:,i]))/np.std(test_x[:,i])
        

        ## Initialization of parameters
        b_h = np.zeros([1,1])
        b_o = np.zeros([3,1])
                    
        W_hh = np.expand_dims(self.initial_weights[0,0:4], axis=1).T
        W_oh = np.expand_dims(self.initial_weights[0,4:7], axis=1)
             
        for lesson in range(self.lessons):
            W_hh, W_oh, b_h, b_o = self.update(W_hh, W_oh, b_h, b_o, train_x, train_y)
            
        test_acc, test_err = self.forward(test_x, test_y, W_hh, W_oh, b_h, b_o)
        
        self.final_weights = [W_hh, W_oh, b_h, b_o]
        
        return test_acc
   
        
        
        
    
    