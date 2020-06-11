#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 12:24:25 2020

@author: Vicky

Custom Training Benchmark -- Nonlinear Regression using Tensorflow 1.X
"""

# %%

import tensorflow as tf 
import numpy as np
from matplotlib import pyplot as plt 
import time 
# %%


def initialize_nn(layers):
    weights = []
    biases = []
    num_layers = len(layers) 
    for l in range(0,num_layers-1):
        W = xavier_init(size=[layers[l], layers[l+1]])
        b = tf.Variable(tf.zeros([1,layers[l+1]]), dtype=tf.float32)
        weights.append(W)
        biases.append(b)        
    return weights, biases
    
def xavier_init(size):
    in_dim = size[0]
    out_dim = size[1]        
    xavier_stddev = np.sqrt(2/(in_dim + out_dim))
    return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

def neural_net(X, weights, biases):
    num_layers = len(weights) + 1
    H = X
    for l in range(0,num_layers-2):
        W = weights[l]
        b = biases[l]
        H = tf.tanh(tf.add(tf.matmul(H, W), b))
    W = weights[-1]
    b = biases[-1]
    Y = tf.add(tf.matmul(H, W), b)
    return Y



class Regressor():
    def __init__(self, layers):
        
            
        self.X = tf.placeholder(tf.float32, shape=[None, 2])
        self.Y = tf.placeholder(tf.float32, shape=[None, 1])
        
        self.X_ic = tf.placeholder(tf.float32, shape=[None, 2])
        self.Y_ic = tf.placeholder(tf.float32, shape=[None,1])
        
        self.weights, self.biases = initialize_nn(layers)
        
        self.Y_pred = self.forward(self.X)
        
        self.loss_ic = tf.reduce_mean(tf.square(self.forward(self.X_ic) - self.Y_ic))
        self.loss_pde = tf.reduce_mean(tf.square(self.pde_loss(self.X)))
        
        #        self.loss_recon = tf.reduce_mean(tf.square(self.Y_pred - self.Y))

        
        self.loss = self.loss_ic + self.loss_pde
        
        self.optimizer = tf.train.AdamOptimizer()
        self.train_ops = self.optimizer.minimize(self.loss)
        
        self.optimizer_QN = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                        method = 'L-BFGS-B', 
                                                        options = {'maxiter': 50000,
                                                                   'maxfun': 50000,
                                                                   'maxcor': 50,
                                                                   'maxls': 50,
                                                                   'ftol' : 1.0 * np.finfo(float).eps})
        
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        
        
        
    def forward(self, X):
        return neural_net(X, self.weights, self.biases)
    
    def pde_loss(self, X):
        u = neural_net(X, self.weights, self.biases)
        u_X = tf.gradients(u, X)[0]
        u_XX = tf.gradients(u_X, X)[0]
        
        loss = u + u_X[:, 1:2] +  4*u_XX[:, 0:1] 
        return tf.reduce_mean(tf.square(loss))
    
    def callback(self, iteration, loss):
        print("Iter: {}, Loss: {}".format(iteration, loss))
        
    def train(self, nIter, train_dict):
        
        train_dict = {self.X: train_dict['X'], self.Y: train_dict['Y'],
                      self.X_ic: train_dict['X_ic'], self.Y_ic: train_dict['Y_ic']}
        
        start_time = time.time()
        
        for it in range(nIter):
            loss_value = self.sess.run(self.loss, train_dict)
            self.sess.run(self.train_ops, train_dict)
            self.callback(it, loss_value)
            
            

        self.optimizer_QN.minimize(self.sess,  
                    feed_dict = train_dict,         
                    fetches = [self.loss])   
            
        end_time = time.time()
        
        print("Total Training Time : {}".format(end_time - start_time))
        
    def predict(self, X):
        return self.sess.run(self.Y_pred, {self.X: X})
        
        
        
if __name__=="__main__":
    
    # Preparing the Dataset 
    N = 5000
    x = np.linspace(-np.pi, np.pi, N)
    t = np.linspace(0, 2, N)
    
    lb = np.asarray([x.min(), t.min()])
    ub = np.asarray([x.max(), t.max()])
    
    def func_u(x, t): # Function that we are interested in modelling.
        return np.sin(x/2) + np.exp(-t)
    
    u_actual = func_u(x, t) 
     
    X = np.vstack((x, t)).T 
    Y = np.reshape(u_actual, (N, 1))
    
    X_ic = np.vstack((x, np.zeros(N))).T
    u_ic = func_u(x, np.zeros(N))
    Y_ic = np.reshape(u_ic, (N,1))
    
    train_dict = {'X': X,
                  'Y': Y,
                  'X_ic': X_ic,
                  'Y_ic': Y_ic}
    
    layers = [2, 128, 256, 128, 1]
    
    model = Regressor(layers)
    
    model.train(nIter=1000, 
                train_dict=train_dict)
    
    Y_pred = model.predict(X)
    
    
    
    
    
    
    
            
    
        
         
        
         
         
         