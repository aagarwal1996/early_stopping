import sys
import math
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.model_selection import KFold
from sklearn.tree import _tree
from sklearn import tree
from sklearn.neighbors import KNeighborsRegressor

def all_tree_paths(dtree, root_node_id=0):
    """
    Get all the individual tree paths from root node to the leaves
    for a decision tree classifier object [1]_.

    Parameters
    ----------
    dtree : DecisionTreeClassifier object
        An individual decision tree classifier object generated from a
        fitted RandomForestClassifierWithWeights object in scikit learn.

    root_node_id : int, optional (default=0)
        The index of the root node of the tree. Should be set as default to
        0 and not changed by the user

    Returns
    -------
    paths : list
        Return a list containing 1d numpy arrays of the node paths
        taken from the root node to the leaf in the decsion tree
        classifier. There is an individual array for each
        leaf node in the decision tree.
    """

    # Use these lists to parse the tree structure
    children_left = dtree.tree_.children_left
    children_right = dtree.tree_.children_right

    if root_node_id is None:
        paths = []

    if root_node_id == _tree.TREE_LEAF:
        raise ValueError("Invalid node_id %s" % _tree.TREE_LEAF)

    # if left/right is None we'll get empty list anyway
    if children_left[root_node_id] != _tree.TREE_LEAF:
        paths = [np.append(root_node_id, l)
                 for l in all_tree_paths(dtree, children_left[root_node_id]) +
                 all_tree_paths(dtree, children_right[root_node_id])]

    else:
        paths = [[root_node_id]]
    return paths

def convert_tree_paths(tree,tree_paths):
    
    """
    
    Converts tree paths so that path is in terms of the features of the X matrix 
    
    """
    
    converted_paths = []
    for path in tree_paths:
        converted_path = []
        for node in path:
            feat = tree.tree_.feature[node] #feat will be index of feature used
            if(feat < 0): #leaf node
                converted_path.append(-1)
            else:
                converted_path.append(feat) #tree_feats[feat] = original column index of X_train 
        converted_paths.append(converted_path)
    return converted_paths

def get_tree_paths(CART):
    CART_structure = convert_tree_paths(CART,all_tree_paths(CART))
    #CART_structure = all_tree_paths(CART)
    return  set([tuple(y) for y in CART_structure])

def get_model_structure(s):
    correct_structure = []
    for i in range(s):
        path = []
        for j in range(i+1):
            path.append(j)
        path.append(-1)
        correct_structure.append(path)
    correct_structure.append(path) #need to double count last path 
    model_structure= set([tuple(y) for y in correct_structure])
    return model_structure

def check_tree_structure(CART,s):
    model_structure = get_model_structure(s)
    CART_structure = get_tree_paths(CART)
    exact_recovery_indicator = 0.0
    print("CART Structure:" + str(CART_structure))
    print("model_structure:" + str(model_structure))
    
    if CART_structure == model_structure:
        exact_recovery_indicator = 1.0
    else:
        exact_recovery_indicator = 0.0
    return exact_recovery_indicator

