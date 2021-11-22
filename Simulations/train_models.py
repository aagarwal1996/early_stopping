import sys
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from generate_data import *

sys.path.append("../")
'''
This script is used to train the following models: 
1.  honest and non-honest CART with min_sample_leaf condition (default = 5)
2.  honest and non-honest CART with CCP with CV 
3.  honest and non-honest CART with early stopping 
4.  KNN 
'''


def CART(X_train,y_train,X_honest,y_honest,X_test,y_test,honest = False):
    if honest == False:
        CART = DecisionTreeRegressor(min_samples_leaf = 5)
        CART.fit(X_train,y_train)
        CART_preds = CART.predict(X_test)
        return mean_squared_error(CART_preds,y_test)
    else:
        CART = DecisionTreeRegressor(min_samples_leaf = 5)
        CART.fit(X_train,y_train)
        honest_test_mse = get_honest_test_MSE(CART,X_honest,y_honest,X_test,y_test)
        return honest_test_mse
        

def CART_CCP(X_train,y_train,X_honest,y_honest,X_test,y_test,sigma,folds = 5):
    id_threshold = sigma**2/len(X_train)
    alphas = np.geomspace(0.1*id_threshold, 1000*id_threshold, num=5)
    scores = []
    models = []
    for alpha in alphas:
        CART = DecisionTreeRegressor(min_samples_leaf = 5,ccp_alpha = alpha)
        CART.fit(X_train,y_train)
        models.append(CART)
        scores.append(cross_val_score(CART, X_train, y_train, cv=folds).mean())
        best_CART = models[scores.index(max(scores))]
        dishonest_MSE = mean_squared_error(best_CART.predict(X_test),y_test)
        #honest_MSE = get_honest_test_MSE(best_CART,X_honest,y_honest,X_test,y_test)
        #return honest_MSE,dishonest_MSE
        return dishonest_MSE
    
def CART_early_stopping(X_train,y_train,X_honest,y_honest,X_test,y_test,sigma):
    id_threshold = (max(sigma**2,1))/(len(X_train))
    CART_early_stopping = DecisionTreeRegressor(min_impurity_decrease = id_threshold)
    CART_early_stopping.fit(X_train,y_train)
    return mean_squared_error(CART_early_stopping.predict(X_test),y_test)
    
    
    
def KNN(X_train,y_train,X_test,y_test,folds):
    knn_regressor = KNeighborsRegressor()
    num_samples = len(X_train)
    param_grid = {'n_neighbors': np.arange(1, round(3*math.log(num_samples)))}
    knn_gscv = GridSearchCV(knn_regressor, param_grid, cv=5)
    knn_gscv.fit(X_train, y_train)
    optimal_nearest_neighbours = knn_gscv.best_params_['n_neighbors']
    optimal_knn_regressor = KNeighborsRegressor(n_neighbors = optimal_nearest_neighbours)
    optimal_knn_regressor.fit(X_train,y_train)
    knn_mse = mean_squared_error(optimal_knn_regressor.predict(X_test),y_test)
    return knn_mse


    
def train_all_models(X_train,y_train,X_honest,y_honest,X_test,y_test,sigma,folds = 5):
    #honest_CART =  CART(X_train,y_train,X_honest,y_honest,X_test,y_test,honest = True)
    #dishonest_CART = CART(X_train,y_train,X_honest,y_honest,X_test,y_test,honest = False)
    
    CART_MSE = CART(X_train,y_train,X_honest,y_honest,X_test,y_test,honest = False)
    CART_CCP_MSE = CART_CCP(X_train,y_train,X_honest,y_honest,X_test,y_test,sigma,folds)
    CART_early_stopping_MSE = CART_early_stopping(X_train,y_train,X_honest,y_honest,X_test,y_test,sigma)
    knn_mse = KNN(X_train,y_train,X_test,y_test,folds = folds)
    
    #honest_CART_CCP,dishonest_CART_CCP = CART_CCP(X_train,y_train,X_honest,y_honest,X_test,y_test,sigma,k = 5)
    return [CART_MSE,CART_CCP_MSE,CART_early_stopping_MSE,knn_mse]
#,honest_CART_CCP
#,,dishonest_CART_CCP