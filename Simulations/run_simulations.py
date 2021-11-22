import os
import pickle as pkl
from os.path import join as oj

#import dvu
import matplotlib as mpl
import matplotlib.pyplot as plt
from math import ceil
from tqdm import tqdm
import numpy as np
import matplotlib.patches as patches
import sys
from math import log
#from simulations_util import *
from collections import defaultdict
from generate_data import *
from train_models import *
import pickle as pkl

#sys.path.append('..')

# change working directory to project root
#if os.getcwd().split('/')[-1] == 'notebooks':
#    os.chdir('../..')

#from experiments.viz import *
#from experiments import viz

model_to_fit = 'spike'
out_dir = 'results/' + model_to_fit
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']



# choose params
#n_train = [100, 250, 500, 750, 1000, 1500] #,1500,2000,2500]
n_train = [100,250,500,750,1000,1500]
n_test = 500
d = 100
beta = 1
sigma = 0.1
sparsity = [3]
n_avg = 5
seed = 1
folds = 5

# keys end up being saps, cart, rf
scores = defaultdict(list)
error_bar = defaultdict(list)
recovery_prob = defaultdict(list)
recovery_prob_error_bars = defaultdict(list)
np.random.seed(seed)

# This cell's code is used to fit and predict for on linear model varying across
# the number of training samples/sparsity 
for s_num, s in enumerate(sparsity):
    print('s_num', s_num)
    scores_s = defaultdict(list)
    error_bar_s = defaultdict(list)
    recovery_prob_s = defaultdict(list)
    recovery_prob_error_bars_s = defaultdict(list)
    fname = oj(out_dir, f'scores_{s}.pkl')

    if os.path.exists(fname):
        continue
    
    for n in tqdm(n_train):
        scores_s_n = defaultdict(list)
        recovery_prob_s_n = defaultdict(list)
        for j in range(n_avg):
            X_train = sample_boolean_X(n,d)
            X_test = sample_boolean_X(n_test, d)
            if model_to_fit == 'spike':     
                y_train = spike_model(X_train, s, beta, sigma)
                y_test = spike_model(X_test, s, beta, 0)
            else:
                y_train = pyramid_model(X_train, s, beta, sigma)
                y_test = pyramid_model(X_test, s, beta, 0)
                

            #for k, m in zip(['SAPS', 'CART', 'RF'], [SaplingSumRegressor(),
             #                                        DecisionTreeRegressor(min_samples_leaf=5),
             #                                        RandomForestRegressor(n_estimators=100, max_features=0.33)]):
            
                #m.fit(X_train, y_train)
                #preds = m.predict(X_test)
                #scores_s_n[k].append(mean_squared_error(y_test, preds))
            models = ['CART','CART_CCP','CART_early_stopping','KNN']
            model_recovery_models = ['CART_CCP','CART_early_stopping']
            model_errors,model_recovery_probabilities = train_all_models(X_train,y_train,X_train,y_train,X_test,y_test,sigma,s,folds = 5)
            for k,m in zip(models,model_errors):   
                scores_s_n[k].append(m)
            for k,m in zip(model_recovery_models,model_recovery_probabilities):
                recovery_prob_s_n[k].append(m)
        for k in scores_s_n:
            scores_s[k].append(np.mean(scores_s_n[k]))
            error_bar_s[k].append(np.std(scores_s_n[k]))
        for k in recovery_prob_s_n:
            recovery_prob_s[k].append(np.mean(recovery_prob_s_n[k]))
            recovery_prob_error_bars_s[k].append(np.std(recovery_prob_s_n[k]))
            
    
    #save results
    for k in scores_s:
        scores[k].append(scores_s[k])
        error_bar[k].append(error_bar_s[k])
    for k in recovery_prob_s:
        recovery_prob[k].append(recovery_prob_s[k])
        recovery_prob_error_bars[k].append(recovery_prob_error_bars_s[k])
    
    os.makedirs(out_dir, exist_ok=True)
    with open(fname, 'wb') as f:
        pkl.dump((scores, error_bar,recovery_prob,recovery_prob_error_bars), f)