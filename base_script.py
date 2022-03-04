
# Importing required libraries -
import numpy as np
import warnings
import pickle
import category_encoders as ce
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score

warnings.simplefilter('ignore')


# Data pre-processing -
def preprocess(raw_data, drop_cols):

    # Removing unnecessary columns -
    raw_data.drop(columns=drop_cols, inplace=True)
    
    # Dropping duplicate rows -
    raw_data.drop_duplicates(inplace=True)
    
    numerical_data = raw_data.select_dtypes(include='number')
    num_cols = numerical_data.columns
    
    categorical_data = raw_data.select_dtypes(exclude='number')
    cat_cols = categorical_data.columns
    
    # Removing outliers -
    Q1 = raw_data[num_cols].quantile(0.25)
    Q3 = raw_data[num_cols].quantile(0.75)
    IQR = Q3 - Q1
    
    raw_data = raw_data[~((raw_data<(Q1-1.5*IQR))|(raw_data>(Q3+1.5*IQR))).any(axis=1)]
    
    # Encoding the categorical variables -
    ord_encoder = ce.OrdinalEncoder(cols=cat_cols)
    processed_data = ord_encoder.fit_transform(raw_data)
    
    return processed_data


def splitter(processed_data, y_var, split_ratio=[]):   
    
    # Splitting the data into dependent & independent variables -
    X = processed_data.drop(columns=y_var, axis=1)
    y = processed_data[y_var].values
    
    # Performing train-test split -
    train_ratio = split_ratio[0] / 100
    validation_ratio = split_ratio[1] / 100
    test_ratio = split_ratio[2] / 100
    
    X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=1-train_ratio, 
                                                        shuffle=True, random_state=42)
    
    X_val, X_test, y_val, y_test = train_test_split(x_test, y_test, 
                                                    test_size=test_ratio/(test_ratio + validation_ratio), 
                                                    shuffle=True, random_state=42) 
    
    X_train = X_train.values
    X_test = X_test.values
    X_val = X_val.values
    
    return X, y, X_train, X_test, X_val, y_train, y_test, y_val
    

'''
def standardizer(X_train, X_test):

    # Standardizing the data -
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_scaled = np.concatenate([X_train_scaled, X_test_scaled], axis=0)

    return X_scaled, X_train_scaled, X_test_scaled 
'''


def model_training(model_obj, X_train, X_test, X_val,  y_train, y_test, y_val):
    
    model_obj.fit(X_train, y_train)
    y_pred_test = model_obj.predict(X_test)
    y_pred_val = model_obj.predict(X_val)
    print("R2 Score (Test): {:.2f}".format(r2_score(y_test, y_pred_test)))
    print("R2 Score (Validation): {:.2f}".format(r2_score(y_val, y_pred_val)))
    
    return model_obj

    
def cross_validation(model_obj, X, y):
    
    kfold = KFold(n_splits=5)
    accuracy = np.mean(cross_val_score(model_obj, X, y, cv=kfold, scoring='r2', n_jobs=-1)) 
    print("Cross Validation Score: {:.2f}".format(accuracy))
    

def dump_model(model_file, model_obj):
    with open(model_file, 'wb') as f:
        pickle.dump(model_obj, f) 
    
    
    



