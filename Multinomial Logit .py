#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy
import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
from matplotlib import pyplot
import numpy as np

# In[2]:

# Load input data
data = pd.read_csv('Input_data.csv')

# In[3]:

# Exogeneous variable
exogen_list = {'gender': data['Gender']}

# Availability variable
avail_list = {'av_DA': data['av_DA'], 
              'av_SR': data['av_SR'],
              'av_Transit': data['av_Transit'],
              'av_Bicycle': data['av_Bicycle'],
              'av_Walk': data['av_Walk']}

avail_val = np.transpose(list(avail_list.values()))

# Endogenous variable
choice_list = ['DA', 'SR', 'Transit', 'Bicycle', 'Walk']

# Define Parameters
params = { #'asc_DA':      tf.constant(tf.zeros([1,]), dtype=tf.float32, name='asc_DA'),
           'asc_SR':      tf.Variable(tf.zeros([1,]), dtype=tf.float32, name='asc_SR'),
           'asc_Transit': tf.Variable(tf.zeros([1,]), dtype=tf.float32, name='asc_Transit'),
           'asc_Bicycle': tf.Variable(tf.zeros([1,]), dtype=tf.float32, name='asc_Bicycle'),
           'asc_Walk':    tf.Variable(tf.zeros([1,]), dtype=tf.float32, name='asc_Walk'),
           #'gender_DA':   tf.Variable(tf.zeros([1,]), dtype=tf.float32, name='gender_DA'),
           'gender_SR':   tf.Variable(tf.zeros([1,]), dtype=tf.float32, name='gender_SR'),
           'gender_transit': tf.Variable(tf.zeros([1,]), dtype=tf.float32, name='gender_transit'),
           'gender_bike': tf.Variable(tf.zeros([1,]), dtype=tf.float32, name='gender_bike'),
           'gender_walk': tf.Variable(tf.zeros([1,]), dtype=tf.float32, name='gender_walk') 
         }

param_name = params.keys()
param_val =  list(params.values())

x_name = exogen_list.keys()
x_val =  list(exogen_list.values())

# One-hot encode
y_train = np.array(data[choice_list].values.reshape(len(data[choice_list]), len(choice_list)), dtype=np.float32)

# Define the utility functions
def utility_func(x_name, x_val, param_name, param_val):
    
    params  = {k:v for k,v in zip(param_name, param_val)}
    x_train = {i:j for i,j in zip(x_name, x_val)}
    
    v1 = tf.zeros([len(x_val[0]),1]) # Baseline
#     v1 = params['gender_DA']*x_train['gender']
    v2 = params['asc_SR'] + params['gender_SR']*x_train['gender']
    v3 = params['asc_Transit'] + params['gender_transit']*x_train['gender']
    v4 = params['asc_Bicycle'] + params['gender_bike']*x_train['gender']
    v5 = params['asc_Walk'] + params['gender_walk']*x_train['gender']
    
#     v1 = tf.reshape(v1, shape=[len(v1), 1])
    v2 = tf.reshape(v2, shape=[len(v2), 1])
    v3 = tf.reshape(v3, shape=[len(v3), 1])
    v4 = tf.reshape(v4, shape=[len(v4), 1])
    v5 = tf.reshape(v5, shape=[len(v5), 1])
    
    return v1, v2, v3, v4, v5

# In[4]:

def model_fun(x_name, x_val, param_name, param_val, avail_val):
    
    # Calculate exponential terms with the availability
    exp_inv = tf.reshape(tf.transpose(tf.exp(utility_func(x_name, x_val,param_name, param_val))), shape=(len(avail_val), np.size(avail_val,1)))
    ex_v_av = avail_val*exp_inv
    exp_summ = tf.reshape(tf.reduce_sum(ex_v_av, axis=1), shape=(len(avail_val), 1))

    # Calculate probability
    P = ex_v_av/exp_summ
    
    return P

# multi-class cost_entropy
def cost_fun(y_train, yhat):
    
    '''The arbitrary value is added to resolve log(zero)'''
    return -tf.reduce_mean(tf.reduce_sum(y_train*tf.math.log(yhat+1e-8), axis=1), axis=0)

"""
To create the gradients of the log-likelihood with respect to parameters, 
we edited the code originally written by Pi-Yueh Chuang <pychuang@gwu.edu>
"""
# obtain the shapes of all trainable parameters in the model
def LL_gradient(x_name, x_val, param_name, param_val, y_train):
    
    shapes = tf.shape_n(param_val)
    n_tensors = len(shapes)

    count = 0
    idx =  [] 
    part = []

    for i, shape in enumerate(shapes):
        n = numpy.product(shape)
        idx.append(tf.reshape(tf.range(count, count+n, dtype=tf.int32), shape))
        part.extend([i]*n)
        count += n
  
    @tf.function
    def assign_new_model_parameters(params_1d):

        updated_params = tf.dynamic_partition(params_1d, part, n_tensors)

        for i, (shape, param) in enumerate(zip(shapes, updated_params)):
            param_val[i].assign(tf.reshape(param, shape))
                   
    @tf.function 
    def est_grad(params_1d):
        
        # Derive the Tensorflow gradient
        with tf.GradientTape() as tape: 
            
            # Call the function to update and convert the shape of parameters
            assign_new_model_parameters(params_1d)
            
            # Estimated Choice Probability 
            yhat = model_fun(x_name, x_val, param_name, param_val, avail_val)
            
            # Call the cost function
            loss_value = cost_fun(y_train, yhat)
            
        # Calculate the gradient for each parameter
        estimated_grad = tape.gradient(loss_value, param_val)

        grads_1dim = tf.dynamic_stitch(idx, estimated_grad)
        return loss_value, grads_1dim
    
    est_grad.idx = idx
    
    return est_grad

# Define the positions of initial parameters
init_params = tf.dynamic_stitch(LL_gradient(x_name, x_val, param_name, param_val, y_train).idx, param_val)

# Implement the BFGS optimizer
Trained_Results = tfp.optimizer.bfgs_minimize(
                                      value_and_gradients_function=LL_gradient(x_name, x_val, param_name, param_val, y_train), 
                                      initial_position=init_params,
                                      tolerance=1e-08,
                                      max_iterations=500)


# In[5]:

# Estimated Variable
est_title = pd.DataFrame(params.keys(), columns=['Variable'])
# Estimated Parameters
est_para = pd.DataFrame(Trained_Results.position.numpy(), columns=['Coef.'])
# Standard Errors
Std_err = pd.DataFrame(np.sqrt(np.diag(pd.DataFrame(Trained_Results.inverse_hessian_estimate.numpy())))/np.sqrt(len(y_train)), columns=['Std.err'])
# t-ratio
t_ratio = pd.DataFrame(est_para.values/Std_err.values, columns=['t-ratio'])
# Estimation results table
Est_result = pd.concat([est_title, est_para, Std_err, t_ratio], axis=1).set_index('Variable')
print(Est_result)

# Loglikelihood Function
LL_initi = tf.reduce_sum(y_train*tf.math.log(model_fun(x_name, x_val, param_name, init_params, avail_val)+1e-8))
LL_final = tf.reduce_sum(y_train*tf.math.log(model_fun(x_name, x_val, param_name, param_val, avail_val)+1e-8))
print("LL(initial):", LL_initi.numpy())
print("LL(final):  ", LL_final.numpy())

# Akaike information criterion (AIC)
Estimated_parameters = len(param_name)
AIC = -2*LL_final+ 2*Estimated_parameters
print("AIC:        ", AIC.numpy())
# Bayesian information criterion (BIC)
BIC = -2*LL_final+ Estimated_parameters*np.log(len(x_val))
print("BIC:        ", BIC.numpy())