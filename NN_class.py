#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 21:23:02 2021

@author: paul
"""

from autograd import grad 
import autograd.numpy as np
from sklearn.model_selection import train_test_split


class NN(object):
    
    def __init__(self, learning_rate, lessons, activation='sigmoid', hidden_layer_size=1, bias=True):
        self.learning_rate = learning_rate
        self.lessons = lessons
        self.act = activation   
        self.hidden_layer_size = hidden_layer_size
        self.bias = bias
        return
    
    def activation(self, x):
        if self.act == 'sigmoid':
            return 1/(np.exp(-x) + 1) 
        if self.act == 'tanh':
            return np.tanh(x)
        if self.act == 'relu':
            return x * (x > 0)
    
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
        num_wrong = np.count_nonzero(np.expand_dims(np.argmax(y, axis=1), axis=1) - pred)
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
        
        if self.bias:
            b_h = b_h - self.learning_rate*dE_dbh
            b_o = b_o - self.learning_rate*dE_dbo          
        return W_hh, W_oh, b_h, b_o
    
    def train_test(self, x, y):
        
        self.initial_weights = np.random.rand(self.hidden_layer_size,len(x[1])+len(y[1])) 
        self.final_weights = []     
        
        # # Split the data for training and testing
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.20, random_state=56)

        #Standardization
        for i in range(len(train_x[0])):
            if np.std(test_x[:,i])!=0 and np.std(train_x[:,i])!=0:
                test_x[:,i] = (test_x[:,i] - np.mean(test_x[:,i]))/np.std(test_x[:,i])
                train_x[:,i] = (train_x[:,i] - np.mean(train_x[:,i]))/np.std(train_x[:,i])
            
        ## Initialization of parameters
        b_h = np.zeros([self.hidden_layer_size,1])
        b_o = np.zeros([len(y[1]),1])
        
        if len(self.initial_weights) == 1:
            W_hh = np.expand_dims(self.initial_weights[0,0:len(x[1])], axis=0)
            W_oh = np.expand_dims(self.initial_weights[0,len(x[1]):len(x[1])+len(y[1])], axis=1)
            
        else:
            W_hh = self.initial_weights[:,0:len(x[1])]
            W_oh = self.initial_weights[:,len(x[1]):len(x[1])+len(y[1])].T
             
        for lesson in range(self.lessons):
            W_hh, W_oh, b_h, b_o = self.update(W_hh, W_oh, b_h, b_o, train_x, train_y)
            
        test_acc, test_err = self.forward(test_x, test_y, W_hh, W_oh, b_h, b_o)
        
        self.final_weights = [W_hh, W_oh, b_h, b_o]
        
        return test_acc
   
        
        
        
    
    