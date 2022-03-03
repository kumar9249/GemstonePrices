
# Importing required libraries -
import pandas as pd
import warnings
from sklearn.tree import DecisionTreeRegressor
import base_script as base

warnings.simplefilter('ignore')


if __name__ == '__main__':
    
    # Loading the dataset -
    data = pd.read_csv(r'D:\Projects\GemstonePrices\cubic_zirconia.csv')
    
    processed_data = base.preprocess(data, 
                                     drop_cols=['Unnamed: 0', 'depth'])
    
    X, y, X_train, X_test, y_train, y_test = base.splitter(processed_data, 
                                                           y_var='price', split_ratio=0.20)
    
    # Decision Tree
    d_tree = DecisionTreeRegressor()
    
    obj = base.model_training(d_tree, X_train, X_test, y_train, y_test)
    base.cross_validation(d_tree, X, y)

    filename = r'tree_model.pkl'
    base.dump_model(filename, obj)

