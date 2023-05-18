import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def X_columnTransformer_OHE(df,target,train_split,is_regression):
    """
        This function takes a dataframe, a target column, a split percentage and a cross validation number.
        It returns a col_transformer, the transformed training and test data, the training and test target, and the prediction array.
        
        Parameters
        ----------
        df : pandas dataframe
            The dataframe to be used for the pipeline.
        target : string
            The target column to be used for the pipeline.
        split : float
            The percentage of the data to be used for testing.
        cross_validation : int
            The number of folds to be used for cross validation.
            
        Returns
        -------
        col_transformer : preprocesador
        X_train_transformed : 
        X_test_transformed : 
        y_train :
        y_test :
        prediction_array :
        
    """
    
    INPUTS = np.setdiff1d(df.columns,target)
    OUTPUT = target
    
    X = pd.DataFrame(df[INPUTS])
    y = pd.DataFrame(df[OUTPUT])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=train_split, 
                                                    random_state=0) 
    
    
    INPUTS_NUM = X.select_dtypes(include=['int64','float64']).columns.values.tolist()
    INPUTS_CAT = X.select_dtypes(include=['object']).columns.values.tolist()

    col_transformer = ColumnTransformer(transformers=[
        ('num', StandardScaler(), INPUTS_NUM),
        ('cat', OneHotEncoder(handle_unknown='ignore',drop="first"), INPUTS_CAT)
        ])

    col_transformer.fit(X_train)
    X_train_transformed = col_transformer.transform(X_train)
    X_test_transformed = col_transformer.transform(X_test) 
    
    cols_X = col_transformer.get_feature_names_out()

    if not is_regression:
        pred_dict_map = {}
        target_encoder = OneHotEncoder(handle_unknown='ignore')
        target_encoder.fit(y)
        y_train_ohe = target_encoder.transform(y_train).toarray()
        y_test_ohe = target_encoder.transform(y_test).toarray()
        #y_val_ohe = target_encoder.transform(y_val).toarray()
        
               
        for i,cat in enumerate(list(target_encoder.categories_[0])):
            pred_dict_map[i] = cat
            
        return (col_transformer,
            X_train_transformed, 
            X_test_transformed, 
            X_train,
            X_test,
            y_train_ohe,
            y_test_ohe,
            pred_dict_map,
            cols_X)
    else: # regresión

        return (col_transformer,
                X_train_transformed, 
                X_test_transformed, 
                X_train,
                X_test,
                y_train.values.ravel(),
                y_test.values.ravel(),
                cols_X)

