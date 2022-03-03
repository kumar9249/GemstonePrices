
# Importing required libraries -
import pandas as pd
import warnings
from sklearn.linear_model import LinearRegression
import base_script as base

warnings.simplefilter('ignore')
    
    
if __name__ == '__main__':
    
    # Loading the dataset -
    data = pd.read_csv(r'D:\Projects\GemstonePrices\cubic_zirconia.csv')
    
    processed_data = base.preprocess(data, 
                                     drop_cols=['Unnamed: 0', 'depth'])
    
    X, y, X_train, X_test, y_train, y_test = base.splitter(processed_data, 
                                                           y_var='price', split_ratio=0.20)
    
    # Linear Regression
    lin_reg = LinearRegression()
    
    obj = base.model_training(lin_reg, X_train, X_test, y_train, y_test)
    base.cross_validation(lin_reg, X, y)
    
    print("\n")
    
    print("Coefficient: ", lin_reg.coef_)
    print("Intercept: ", lin_reg.intercept_)

    filename = r'lr_model.pkl'
    base.dump_model(filename, obj)
    



