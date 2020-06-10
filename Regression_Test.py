#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 12:24:25 2020

@author: Vicky

Custom Training Benchmark -- Nonlinear Regression 
"""

# %%
import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow import keras
tf.keras.backend.set_floatx('float64')
tf.random.set_seed(0)

import torch
torch.manual_seed(0)

# %%
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
# X = (X-lb) / (ub  - lb)

# %%
#Tensorflow Model

def tf_model(): # Creating a two layer Neural Network. 
    act_func = 'relu'
    model_tf = keras.Sequential()
    model_tf.add(keras.layers.Input(shape=(2,)))
    model_tf.add(keras.layers.Dense(100, activation=act_func))
    model_tf.add(keras.layers.Dense(100, activation=act_func))
    model_tf.add(keras.layers.Dense(1, activation='linear'))
    
    return model_tf


model = tf_model()
model.compile(optimizer='adam', loss='mse')

model.fit(X, u_actual, # Fitting using the Keras API
          batch_size=500,
          epochs=1000,
          verbose=1)

u_model_fit = model(X).numpy()


# %%

def shuffle_and_batch(X, Y, num_batches=50): # Shuffle and group the input data inot various datasizes. 
    indices = tf.range(start=0, limit=tf.shape(X)[0], dtype=tf.int32) 
    shuffled_indices = tf.random.shuffle(indices)
    
    X = tf.gather(X, shuffled_indices)
    Y = tf.gather(Y, shuffled_indices)

    X = tf.split(X, num_batches)
    Y = tf.split(Y, num_batches)
    return X, Y


X_tf = tf.Variable(X, tf.float64) #Creating Tensors to treat as the inputs. 
Y_tf = tf.Variable(u_actual, tf.float64)

X_tf, Y_tf = shuffle_and_batch(X_tf, Y_tf)


def loss(model, X, Y): # mean squared reconstruction error 
    return tf.reduce_mean(tf.square(model(X, training=True) - Y))
                          
def loss_and_gradients(model, X, Y):
    with tf.GradientTape() as tape:
        loss_tf = loss(model, X, Y)
    grads_tf = tape.gradient(loss_tf, model.trainable_variables)  #Calculating the gradient of each of the loss with respect to the weights and biases. 
    return loss_tf, grads_tf
    

optimizer = tf.keras.optimizers.Adam()
nIter =1000
model = tf_model()

for it in range(nIter):
    for batch_num in range(50):
        loss_tf, grads_tf = loss_and_gradients(model, X_tf[batch_num], Y_tf[batch_num]) #Obtaining the loss and the gradients
        optimizer.apply_gradients(zip(grads_tf, model.trainable_variables)) #Applying the gradients for each step 

    tf.print('Iter : {}, loss : {}'.format(it, loss_tf))
    
u_tf = model(X).numpy()
    


# %%

model_torch = torch.nn.Sequential(
    torch.nn.Linear(2, 100),
    torch.nn.ReLU(),
    torch.nn.Linear(100, 100),
    torch.nn.ReLU(),
    torch.nn.Linear(100, 1)
    )

X_torch = torch.tensor(X, dtype=torch.float64).float()
Y_torch = torch.tensor(u_actual, dtype=torch.float64).float()

# X_torch, Y_torch = shuffle_and_batch_torch(X_torch, Y_torch)
X_torch, Y_torch = torch.autograd.Variable(X_torch, requires_grad=True), torch.autograd.Variable(Y_torch, requires_grad=True) #Ensuring that tracing occurs. 

traindata = torch.utils.data.TensorDataset(X_torch, Y_torch) #Loading, Shuffling and Batching the training data. 
dataloader = torch.utils.data.DataLoader(traindata, batch_size=50, shuffle=True)

def lossfunc_torch(model, X, Y): # Calculating the mean squared error, 
    y = model_torch(X)    
    loss = (y-Y).pow(2).mean()
    
    return loss

optimizer = torch.optim.Adam(model_torch.parameters(), 0.001)    
nIter = 1000


for it in range(nIter):
    for i, (x_torch, y_torch) in enumerate (dataloader):
        optimizer.zero_grad()
        
        loss_torch = lossfunc_torch(model_torch, x_torch, y_torch)
    
        loss_torch.backward()
        optimizer.step()
        
        
    print("Iter : {}, Loss : {}".format(it, loss_torch.item()))


u_torch = model_torch(X_torch).detach().numpy()

# %%
#Plotting the regression fits for each approach
plt.figure()
plt.plot(u_actual, label='Actual')
plt.plot(u_model_fit, label='Keras Fit')
plt.plot(u_tf, label='TF GradientTape')
plt.plot(u_torch, label='PyTorch')
plt.legend()