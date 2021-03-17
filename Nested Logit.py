#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy
import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
import numpy as np
import timeit

# In[2]:

# Load input data
data = pd.read_csv('Input_data.csv')

# Set global values
global non_zero
non_zero = 1e-16

# max and min function

# Exogeneous variable
exogen_list = {'gender': data['Gender'],
               'hhincome_4': data['hhincome_4'],
               'graduate': data['Graduate'],
               'hhsize_3GE': data['hhsize3GE'],
               'age25_29': data['age25_29'],
               'age30_44': data['age30_44'],
               'age45_59': data['age45_59'],
               'log_triptime': data['triptime_log'],
               'bachelor': data['Bachelor'],
               'emply_st': data['Employ_status'],
               'Ur_ru': data['Urban_Rural'],
               'constant': data['constant']}

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
           'lambda_coff': tf.Variable([0.95,], dtype=tf.float32, name='lambda_coff'),    
           'gender_sr': tf.Variable(tf.zeros([1,]), dtype=tf.float32, name='gender_sr')}

param_name = params.keys()
param_val =  list(params.values())

x_name = exogen_list.keys()
x_val =  list(exogen_list.values())

# One-hot encode
y_train = np.array(data[choice_list].values.reshape(len(data[choice_list]), len(choice_list)), dtype=np.float32)

# In[3]:

# Define the utility functions
def utility_func(x_name, x_val, param_name, param_val):
    
    # Define the utility functions
    params  = {k:v for k,v in zip(param_name, param_val)}
    x_train = {i:j for i,j in zip(x_name, x_val)}

    v1 = 0.0 # Baseline

    v2 = params['asc_SR'] + params['gender_sr']*x_train['gender']

    v3 = params['asc_Transit']

    v4 = params['asc_Bicycle']

    v5 = params['asc_Walk']

    v = [v1, v2, v3, v4, v5]

    for i in range(len(v)):
        if tf.size(v[i])>1:
            v[i] = tf.reshape(v[i], shape=[tf.size(v[i]), 1])

    v = {'DA':v[0], 'SR':v[1], 'Transit':v[2], 'Bicycle':v[3], 'Walk':v[4]}
    
    return v

# Define the nest structure
def nested_structure():
    # Two-level nest
    upper_level = ['DA','SR','Transit']
    
    nest_level = ['Bicycle', 'Walk']
    
    return upper_level, nest_level

# In[4]:

def model_fun(x_name, x_val, param_name, param_val, avail_list):
    
    params  = {k:v for k,v in zip(param_name, param_val)}

    # nest-level
    nestd_coeff = params['lambda_coff']
    n = []
    for i in nested_structure()[1]:
        exp_inv = tf.cast(tf.reshape(avail_list['av_'+i], shape=[len(avail_list['av_'+i]), 1]), dtype=tf.float32)*tf.exp(utility_func(x_name, x_val, param_name, param_val)[i]/nestd_coeff)
        n.append(exp_inv)

    # Generate the matrix form of the nested component
    nested_mat = tf.concat(n, axis=1)
    nest_exp_sum = tf.reshape(tf.reduce_sum(nested_mat, axis=1), shape=(len(nested_mat), 1))

    # Define the logsum
    log_sum = tf.math.log(nest_exp_sum + non_zero)

    # Conditional probability of the nested components
    condi_prob = nested_mat/(nest_exp_sum + non_zero)
    
    # upper-level
    U = []
    for i in nested_structure()[0]:
        exp_inv = tf.cast(tf.reshape(avail_list['av_'+i], shape=[len(avail_list['av_'+i]), 1]), dtype=tf.float32)*tf.exp(utility_func(x_name, x_val, param_name, param_val)[i])
        U.append(exp_inv)

    upper_mat = tf.concat(U, axis=1)
    upper_exp_sum = tf.reshape(tf.reduce_sum(upper_mat, axis=1), shape=(len(upper_mat), 1))

    # Upper level probability
    upper_prob = upper_mat/(upper_exp_sum + tf.exp(nestd_coeff*log_sum))    
    
    
    # Marginal probability
    n_marginal = tf.exp(nestd_coeff*log_sum)/(upper_exp_sum + tf.exp(nestd_coeff*log_sum))

    # Joint probability
    p_joint = n_marginal*condi_prob
    
    # Likelihood
    P = tf.concat([upper_prob, p_joint], axis=1)
    
    return P

# In[5]:

# multi-class cost_entropy
def cost_fun(y_train, yhat):
    
    '''The arbitrary value, non_zero, is added to resolve log(zero)'''
    return -tf.reduce_mean(tf.reduce_sum(y_train*tf.math.log(yhat + non_zero), axis=1), axis=0)

"""
To create the gradients of variables, 
we edited the code written by Pi-Yueh Chuang <pychuang@gwu.edu>
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
            yhat = model_fun(x_name, x_val, param_name, param_val, avail_list)
            
            # Call the cost function
            loss_value = cost_fun(y_train, yhat)
            
        # Calculate the gradient for each parameter
        estimated_grad = tape.gradient(loss_value, param_val)
        
        grads_1dim = tf.dynamic_stitch(idx, estimated_grad)
        return loss_value, grads_1dim
    
    est_grad.idx = idx
    
    return est_grad

# Define the initial parameters
init_params = tf.dynamic_stitch(LL_gradient(x_name, x_val, param_name, param_val, y_train).idx, param_val)

# the BFGS optimizer
Trained_Results = tfp.optimizer.bfgs_minimize(value_and_gradients_function=LL_gradient(x_name, x_val, param_name, param_val, y_train), 
                                      initial_position=init_params, 
                                      tolerance=1e-10, 
                                      max_iterations=500)

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

# # Loglikelihood Function
LL_initi = tf.reduce_sum(y_train*tf.math.log(model_fun(x_name, x_val, param_name, init_params, avail_list) + non_zero))
LL_final = tf.reduce_sum(y_train*tf.math.log(model_fun(x_name, x_val, param_name, param_val, avail_list) + non_zero))
print("LL(initial):", LL_initi.numpy())
print("LL(final):  ", LL_final.numpy())

# Akaike information criterion (AIC)
Estimated_parameters = len(param_name)
AIC = -2*LL_final+ 2*Estimated_parameters
print("AIC:        ", AIC.numpy())
# Bayesian information criterion (BIC)
BIC = -2*LL_final+ Estimated_parameters*np.log(len(x_val))
print("BIC:        ", BIC.numpy())