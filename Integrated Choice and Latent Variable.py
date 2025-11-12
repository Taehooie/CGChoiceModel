import numpy
import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
import numpy as np
import timeit


# Load Input Data
data = pd.read_csv('ICLV_input.csv')


''' Explanatory Variables for latent variables '''
lat_exo_list = {'regular_user': data['regular_user'], 
                'university_educated': data['university_educated'],
                'over_50': data['over_50']}

''' Explanatory Variables for the utility function '''
uti_exo_list = {'brand': data['brand'], 
                'side_effects_1': data['side_effects_1'],
                'side_effects_2': data['side_effects_2'],
                'side_effects_3': data['side_effects_3'],
                'side_effects_4': data['side_effects_4'],
                'price_1': data['price_1'],
                'price_2': data['price_2'],
                'price_3': data['price_3'],
                'price_4': data['price_4']}

''' Attitude questions for measurement indicators '''
# Centre them on zero to avoid calculating the mean of the normal distribution
data['attitude_quality']     = data['attitude_quality']-np.mean(data['attitude_quality'])
data['attitude_ingredients'] = data['attitude_ingredients']-np.mean(data['attitude_ingredients'])
data['attitude_patent']      = data['attitude_patent']-np.mean(data['attitude_patent'])
data['attitude_dominance']   = data['attitude_dominance']-np.mean(data['attitude_dominance'])

# Attitudinal questions
att_list = {'attitude_quality': data['attitude_quality'],
            'attitude_ingredients': data['attitude_ingredients'],
            'attitude_patent': data['attitude_patent'],
            'attitude_dominance': data['attitude_dominance']}

''' Endogenous (dependent) Variables '''
choice_list = ['best_1', 'best_2', 'best_3', 'best_4']

''' Parameters '''
params = {'b_risk': tf.Variable(tf.zeros([1,]), dtype=tf.float32, name='b_risk'),
          'b_price': tf.Variable(tf.zeros([1,]), dtype=tf.float32, name='b_price'),
          'gamma_reg_user': tf.Variable(tf.zeros([1,]), dtype=tf.float32,   name='gamma_reg_user'),
          'gamma_university': tf.Variable(tf.zeros([1,]), dtype=tf.float32, name='gamma_university'),
          'gamma_age_50': tf.Variable(tf.zeros([1,]), dtype=tf.float32,     name='gamma_age_50'),
          'lambda_coff': tf.Variable(tf.zeros([1,]), dtype=tf.float32, name='lambda'),
          'zeta_qual': tf.Variable([1.0], dtype=tf.float32, name='zeta_qual'),
          'zeta_ingr': tf.Variable([1.0], dtype=tf.float32, name='zeta_ingr'),
          'zeta_pate': tf.Variable([1.0], dtype=tf.float32, name='zeta_pate'),
          'zeta_domi': tf.Variable([1.0], dtype=tf.float32, name='zeta_domi'),
          'sigma_qual': tf.Variable([1.0], dtype=tf.float32, name='sigma_qual'),
          'sigma_ingr': tf.Variable([1.0], dtype=tf.float32, name='sigma_ingr'),
          'sigma_pate': tf.Variable([1.0], dtype=tf.float32, name='sigma_pate'),
          'sigma_domi': tf.Variable([1.0], dtype=tf.float32, name='sigma_domi')
          }

param_name = params.keys()
param_val =  list(params.values())

# Combine dictionaries
com_dict = {**lat_exo_list, **uti_exo_list, **att_list}
x_name = com_dict.keys()
x_val =  list(com_dict.values())

# One-hot encode
y_train = np.array(data[choice_list].values.reshape(len(data[choice_list]), len(choice_list)), dtype=np.float32)


def normal_dist(sigma_unconstrained, x_atti, zeta, latent_val):
    sigma = tf.nn.softplus(sigma_unconstrained) + 1e-6 # positive constraint
    L_normal = (1/(tf.sqrt(2*np.pi)*sigma))*tf.exp(-0.5*((np.array(x_atti)-zeta*latent_val)/(sigma))**2)
    L_normal = tf.reshape(L_normal, shape=[len(L_normal), 1])

    return L_normal


def mnl_func(v):
    
    exp_inv = tf.reshape(tf.transpose(tf.exp(v)), shape=(len(v[0]), len(v)))

    exp_sum = tf.reshape(tf.reduce_sum(exp_inv, axis=1), shape=(len(v[0]), 1))

    mnl_P = exp_inv/exp_sum
    
    return mnl_P


def model_fun(x_name, x_val, param_name, param_val):
    
    # Parameter dictionary
    params = {k:v for k,v in zip(param_name, param_val)}
    
    # Input values dictionary
    x_train = {i:j for i,j in zip(x_name, x_val)}
    
    # Monte Carlo drawing numbers
    sim_step = 500
    
    # Build a matrix for the likelihood
    sll = tf.zeros(shape=[len(y_train), y_train.shape[1]])
    
    # Drawing normally distributed random values
    eta = tf.random.normal([len(y_train), sim_step], 0.0, 1.0, tf.float32)
    eta = tf.reshape(eta, shape=[sim_step, len(y_train)])
    
    # Convert numpy pi values into tensors
    pi_val = tf.constant(np.pi)
    
    for i in eta:

        # Define the latent variable expression
        latent_val = params['gamma_reg_user']*x_train['regular_user']+params['gamma_university']*x_train['university_educated'] + params['gamma_age_50']*x_train['over_50'] + i

        # Likelihood of Indicators
        p_indi_1 = normal_dist(params['sigma_qual'],
                                x_train['attitude_quality'],
                                params['zeta_qual'],
                                latent_val)

        p_indi_2 = normal_dist(params['sigma_ingr'],
                                x_train['attitude_ingredients'],
                                params['zeta_ingr'],
                                latent_val) 

        p_indi_3 = normal_dist(params['sigma_pate'],
                                x_train['attitude_patent'],
                                params['zeta_pate'],
                                latent_val) 

        p_indi_4 = normal_dist(params['sigma_domi'],
                                x_train['attitude_dominance'],
                                params['zeta_domi'],
                                latent_val) 
        
        # Joint probabilities (indicators)
        p_indi = p_indi_1 * p_indi_2 * p_indi_3 * p_indi_4
        
        # Define utility functions with latent variables
        v1 = params['b_price']*x_train['price_1'] + params['lambda_coff']*latent_val

        v2 = params['b_price']*x_train['price_2'] + params['b_risk']*x_train['side_effects_2']+ params['lambda_coff']*latent_val
        
        v3 = params['b_price']*x_train['price_3'] + params['b_risk']*x_train['side_effects_3']

        v4 = params['b_price']*x_train['price_4'] + params['b_risk']*x_train['side_effects_4']
        
        v = [v1, v2, v3, v4]

        # Multinomial Logit
        mnl_p = mnl_func(v)
        
        # Joint Choice Probability
        p = mnl_p * p_indi
        sll = p + sll

    # Likelihood of the simulation-based ICLV    
    sLL = sll/sim_step
    
    return sLL

# Cost function for the multi classification
def cost_fun(y_train, yhat):

    return -tf.reduce_mean(tf.reduce_sum(y_train*tf.math.log(yhat + 1e-8), axis=1), axis=0)

# Obtain the shapes of all trainable parameters in the model (Estimation)
def loss_gradient(x_name, x_val, param_name, param_val, y_train):
    
    shapes = tf.shape_n(param_val)
    n_tensors = len(shapes)
    
    count = 0
    idx = []  # stitch indices
    part = [] # partition indices
    
    for i, shape in enumerate(shapes):
        n = numpy.product(shape)
        idx.append(tf.reshape(tf.range(count, count+n, dtype=tf.int32), shape))
        part.extend([i]*n)
        count += n

    @tf.function
    def assign_new_model_parameters(params_1d):

        pparams = tf.dynamic_partition(params_1d, part, n_tensors)

        for i, (shape, param) in enumerate(zip(shapes, pparams)):
            param_val[i].assign(tf.reshape(param, shape))

    @tf.function 
    def est_grad(params_1d):
        # Derive the Tensorflow gradient
        with tf.GradientTape() as tape: 

            # Call the function to update and convert the shape of parameters
            assign_new_model_parameters(params_1d)
            
            # Estimated Choice Probability
            yhat = model_fun(x_name, x_val, param_name, param_val)
            
            # Call the loss function
            loss_value = cost_fun(y_train, yhat)

        # Calculate the gradient for each parameter
        estimated_grad = tape.gradient(loss_value, param_val)

        grads_1dim = tf.dynamic_stitch(idx, estimated_grad)
        return loss_value, grads_1dim

    est_grad.idx = idx

    return est_grad

# Define the initial parameters
init_params = tf.dynamic_stitch(loss_gradient(x_name, 
                                              x_val, 
                                              param_name, 
                                              param_val, 
                                              y_train).idx, param_val)

# The package "lbfgs" does not provide "inverse_hessian".
Trained_Results = tfp.optimizer.bfgs_minimize(value_and_gradients_function=loss_gradient(x_name, 
                                                                                         x_val, 
                                                                                         param_name, 
                                                                                         param_val, 
                                                                                         y_train), 
                                                                                         initial_position=init_params,
                                                                                         tolerance=1e-08,
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


# Loglikelihood Function
LL_initi = tf.reduce_sum(y_train*tf.math.log(model_fun(x_name, x_val, param_name, init_params)+1e-8))
LL_final = tf.reduce_sum(y_train*tf.math.log(model_fun(x_name, x_val, param_name, param_val)+1e-8))
print("LL(initial):", LL_initi.numpy())
print("LL(final):  ", LL_final.numpy())


# Akaike information criterion (AIC)
Estimated_parameters = len(param_name)
AIC = -2*LL_final+ 2*Estimated_parameters
print("AIC:        ", AIC.numpy())

# Bayesian information criterion (BIC)
BIC = -2*LL_final+ Estimated_parameters*np.log(len(x_val[0]))
print("BIC:        ", BIC.numpy())

