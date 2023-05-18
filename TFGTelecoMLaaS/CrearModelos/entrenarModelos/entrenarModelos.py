import os
import warnings

import keras
import numpy as np
import optuna
import tensorflow as tf
from numpy.random import seed
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import (ElasticNetCV, LinearRegression,
                                  LogisticRegression)
from sklearn.metrics import balanced_accuracy_score, mean_squared_error
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
TOL = 0.5

#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
##################################===CLASIFICACIÓN===#################################################
#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------






######################################################################################################
######################################===KNN===#######################################################
######################################################################################################
def train_knnClass(modelo,n_neighbors,weights,metric):
    (X_train,y_train) = modelo.get_training_data()
    model = KNeighborsClassifier(n_neighbors=n_neighbors,weights=weights,metric=metric)
    model.fit(X_train,y_train)
    return model

def train_knnClass_optuna(modelo):
    (X_train,y_train) = modelo.get_training_data()
    def objective(trial):
        n_neighbors = trial.suggest_int("n_neighbors", 3, 40)
        weights = trial.suggest_categorical("weights", ['uniform', 'distance'])
        metric = trial.suggest_categorical("metric", ['euclidean', 'manhattan', 'minkowski'])
        knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric)

        # -- Cross-validate the features reduced by dimensionality reduction methods
        kfold = StratifiedKFold(n_splits=10,)
        score = cross_val_score(knn, X_train, y_train.argmax(axis=1), scoring='balanced_accuracy', cv=kfold)
        score = score.mean()
        return score
    
    sampler = TPESampler(seed=0) # create a seed for the sampler for reproducibility
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=20)
    
    model = KNeighborsClassifier(n_neighbors=study.best_params["n_neighbors"],
                                 weights=study.best_params["weights"],
                                 metric=study.best_params["metric"])
    model.fit(X_train,y_train)
    return (model,study)
    
    
    
    
    
    
######################################################################################################
######################################===SVC===#######################################################
######################################################################################################
    
def train_SVC(modelo,kernel,C,gamma): # DOESNT SUPPORT ONE HOT ENCODING in Y
    (X_train,y_train) = modelo.get_training_data()
    model = SVC(C=C,gamma=gamma,kernel=kernel,probability=True)
    model.fit(X_train,y_train)
    return model

def train_SVC_optuna(modelo):
    (X_train,y_train) = modelo.get_training_data()
    def objective(trial):
        kernel = trial.suggest_categorical("kernel", ['poly', 'rbf','sigmoid'])
        C = trial.suggest_float("C",0.001,100)
        gamma = trial.suggest_categorical("gamma", ['scale', 'auto'])
        svc = SVC(kernel=kernel,C=C,gamma=gamma)

        # -- Cross-validate the features reduced by dimensionality reduction methods
        kfold = StratifiedKFold(n_splits=10)
        score = cross_val_score(svc, X_train, y_train, scoring='balanced_accuracy', cv=kfold)
        score = score.mean()
        return score
    
    sampler = TPESampler(seed=0) # create a seed for the sampler for reproducibility
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=7)
    
    model = SVC(kernel=study.best_params["kernel"],
                C=study.best_params["C"],
                gamma=study.best_params["gamma"],
                probability=True)
    model.fit(X_train,y_train)
    return (model,study)


######################################################################################################
######################################===Regresión Logística===#######################################
######################################################################################################
    
def train_LogisticRegression(modelo,solver,C,penalty): 
    (X_train,y_train) = modelo.get_training_data()
    multi_class = "auto"
    class_weight = "balanced"
    if not modelo.is_binary_model():
        multi_class = "ovr"
    model = LogisticRegression(solver=solver,C=C,penalty=penalty,multi_class=multi_class,class_weight=class_weight)
    model.fit(X_train,y_train)
    return model

def train_LogisticRegression_optuna(modelo):
    (X_train,y_train) = modelo.get_training_data()
    multi_class = "auto"
    class_weight = "balanced"
    if not modelo.is_binary_model():
        multi_class = "ovr"
        #class_weight = "balanced"
    def objective(trial):
        solver = trial.suggest_categorical("solver", ['lbfgs','liblinear', 'saga'])
        C = trial.suggest_float("C",0.001,10)
        penalty = trial.suggest_categorical("penalty", ['l1', 'l2'])
        
        model = LogisticRegression(solver=solver,C=C,penalty=penalty,multi_class=multi_class,class_weight=class_weight,max_iter=300)

        # -- Cross-validate the features reduced by dimensionality reduction methods
        kfold = StratifiedKFold(n_splits=10)
        score = cross_val_score(model, X_train, y_train, scoring='balanced_accuracy', cv=kfold)
        score = score.mean()
        return score
    
    sampler = TPESampler(seed=0) # create a seed for the sampler for reproducibility
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=20,n_jobs=2)
    
    model = LogisticRegression(solver=study.best_params["solver"],
                                C=study.best_params["C"],
                                penalty=study.best_params["penalty"],
                                multi_class=multi_class,
                                class_weight=class_weight,max_iter=500)
    model.fit(X_train,y_train)
    return (model,study)



######################################################################################################
######################################===Random Forest===#############################################
######################################################################################################

def train_RandomForestClass(modelo,n_estimators,criterion,max_depth,min_samples_leaf):
    (X_train,y_train) = modelo.get_training_data()
    
    model = RandomForestClassifier(n_estimators = n_estimators,criterion = criterion,max_depth = max_depth,min_samples_leaf = min_samples_leaf)
    model.fit(X_train,y_train)
    return model

def train_RandomForestClass_optuna(modelo):
    (X_train,y_train) = modelo.get_training_data()
    print("Estoy entrenado un Random Forest")

    def objective(trial):
        criterion = trial.suggest_categorical("criterion", ["gini","entropy"])
        n_estimators = trial.suggest_int("n_estimators",50,5000)
        max_depth = trial.suggest_int("max_depth",150,1000)
        min_samples_leaf = trial.suggest_int("min_samples_leaf",1,15)
        
        model = RandomForestClassifier(n_estimators = n_estimators,criterion = criterion,max_depth = max_depth,min_samples_leaf = min_samples_leaf)

        # -- Cross-validate the features reduced by dimensionality reduction methods
        kfold = StratifiedKFold(n_splits=10)
        score = cross_val_score(model, X_train, y_train.argmax(axis=1), scoring='balanced_accuracy', cv=kfold)
        score = score.mean()
        return score
    
    sampler = TPESampler(seed=0) # create a seed for the sampler for reproducibility
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=12,n_jobs=3)
    
    model = RandomForestClassifier(n_estimators=study.best_params["n_estimators"],
                                    criterion=study.best_params["criterion"],
                                    max_depth=study.best_params["max_depth"],
                                    min_samples_leaf=study.best_params["min_samples_leaf"])
    model.fit(X_train,y_train)
    return (model,study)





#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
##################################===REGRESIÓN===#####################################################
#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------


######################################################################################################
##################################===ElasticNetCV===##################################################
######################################################################################################

def train_ElasticNetCV(modelo,l1_ratio,eps,n_alphas):
    (X_train,y_train) = modelo.get_training_data()
    
    model = ElasticNetCV(l1_ratio=l1_ratio,
                        eps=eps,
                        n_alphas=n_alphas,
                        tol=TOL)
    model.fit(X_train,y_train)
    return model

def train_ElasticNetCV_optuna(modelo):
    (X_train,y_train) = modelo.get_training_data()
    
    def objective(trial):
        n_alphas = trial.suggest_int("n_alphas",50,200)
        eps = trial.suggest_float("eps",0.001,0.1)
        l1_ratio = trial.suggest_float("l1_ratio", 0.0,1.0)
        model = ElasticNetCV(l1_ratio=l1_ratio,
                                      eps=eps,
                                      max_iter = 4000,
                                      n_alphas=n_alphas,
                                      #tol=TOL,
                                      cv = 10,
                                      random_state=7
                                      )
        
        X_train_2, X_val, y_train_2, y_val = train_test_split(X_train, y_train, test_size=0.2)

        model.fit(X_train_2,y_train_2)
        score = -float(mean_squared_error(y_true=y_val,y_pred= model.predict(X_val)))
        return score
    
    sampler = TPESampler(seed=0) # create a seed for the sampler for reproducibility
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=40,n_jobs=2)
    model = ElasticNetCV(l1_ratio=study.best_params["l1_ratio"],
                        eps=study.best_params["eps"],
                        max_iter = 2000,
                        n_alphas=study.best_params["n_alphas"],
                        #tol=TOL,
                        random_state=7
                        )
    model.fit(X_train,y_train)
    return (model,study)

######################################################################################################
######################################===Random Forest===#############################################
######################################################################################################

def train_RandomForestReg(modelo,n_estimators,criterion,max_depth,min_samples_leaf):
    (X_train,y_train) = modelo.get_training_data()
    
    model = RandomForestRegressor(n_estimators = n_estimators,criterion = criterion,max_depth = max_depth,min_samples_leaf = min_samples_leaf)
    model.fit(X_train,y_train)
    return model

def train_RandomForestReg_optuna(modelo):
    (X_train,y_train) = modelo.get_training_data()

    def objective(trial):
        criterion = trial.suggest_categorical("criterion", ["squared_error","absolute_error","friedman_mse","poisson"])
        n_estimators = trial.suggest_int("n_estimators",50,5000)
        max_depth = trial.suggest_int("max_depth",15,1000)
        min_samples_leaf = trial.suggest_int("min_samples_leaf",1,50)
        
        model = RandomForestRegressor(n_estimators = n_estimators,
                                      criterion = criterion,
                                      max_depth = max_depth,
                                      min_samples_leaf = min_samples_leaf,
                                      random_state=7)
        X_train_2, X_val, y_train_2, y_val = train_test_split(X_train, y_train, test_size=0.2)

        model.fit(X_train_2,y_train_2)
        score = -float(mean_squared_error(y_true=y_val,y_pred= model.predict(X_val)))
        return score
    
    sampler = TPESampler(seed=0) # create a seed for the sampler for reproducibility
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=9,n_jobs=3)
    
    model = RandomForestRegressor( n_estimators=study.best_params["n_estimators"],
                                    criterion=study.best_params["criterion"],
                                    max_depth=study.best_params["max_depth"],
                                    min_samples_leaf=study.best_params["min_samples_leaf"],
                                    random_state=7)
    model.fit(X_train,y_train)
    print(mean_squared_error(y_true=y_train,y_pred= model.predict(X_train)))
    return (model,study)


######################################################################################################
######################################===KNN===#######################################################
######################################################################################################
def train_knnReg(modelo,n_neighbors,weights,metric):
    (X_train,y_train) = modelo.get_training_data()
    model = KNeighborsRegressor(n_neighbors=n_neighbors,weights=weights,metric=metric)
    model.fit(X_train,y_train)
    return model

def train_knnReg_optuna(modelo):
    (X_train,y_train) = modelo.get_training_data()
    def objective(trial):
        n_neighbors = trial.suggest_int("n_neighbors", 3, 40)
        weights = trial.suggest_categorical("weights", ['uniform', 'distance'])
        metric = trial.suggest_categorical("metric", ['euclidean', 'manhattan', 'minkowski'])
        model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, metric=metric)

        # -- Cross-validate the features reduced by dimensionality reduction methods
        X_train_2, X_val, y_train_2, y_val = train_test_split(X_train, y_train, test_size=0.2)

        model.fit(X_train_2,y_train_2)
        score = -float(mean_squared_error(y_true=y_val,y_pred= model.predict(X_val)))
        return score
    
    sampler = TPESampler(seed=0) # create a seed for the sampler for reproducibility
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=40,n_jobs=3)
    
    model = KNeighborsRegressor(n_neighbors=study.best_params["n_neighbors"],
                                 weights=study.best_params["weights"],
                                 metric=study.best_params["metric"])
    model.fit(X_train,y_train)
    return (model,study)

######################################################################################################
#####################################===Regresion Lineal===###########################################
######################################################################################################

def train_linear_regression(modelo):
    (X_train,y_train) = modelo.get_training_data()
    model = LinearRegression()
    model.fit(X_train,y_train)
    return model
    
######################################################################################################
#####################################===Regresion Lineal===###########################################
######################################################################################################

def train_SVR(modelo,kernel,degree,C,gamma):
    (X_train,y_train) = modelo.get_training_data()
    model = SVR(kernel=kernel,C=C,gamma=gamma,degree=degree)
    model.fit(X_train,y_train)
    return model

def train_SVR_optuna(modelo):
    (X_train,y_train) = modelo.get_training_data()
    def objective(trial):
        kernel = trial.suggest_categorical("kernel", ['poly', 'rbf','sigmoid','linear'])
        degree = trial.suggest_int("degree",1,7)
        C = trial.suggest_float("C",0.001,100)
        gamma = trial.suggest_categorical("gamma", ['scale', 'auto'])
        model = SVR(kernel=kernel,C=C,gamma=gamma,degree=degree)

        X_train_2, X_val, y_train_2, y_val = train_test_split(X_train, y_train, test_size=0.2)

        model.fit(X_train_2,y_train_2)
        score = -float(mean_squared_error(y_true=y_val,y_pred= model.predict(X_val)))
        return score
    
    sampler = TPESampler(seed=0) # create a seed for the sampler for reproducibility
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=20,n_jobs=2)
    
    model = SVR(kernel=study.best_params["kernel"],
                C=study.best_params["C"],
                gamma=study.best_params["gamma"],
                degree=study.best_params["degree"])
    model.fit(X_train,y_train)
    return (model,study)


######################################################################################################
###########################################===MLP===##################################################
######################################################################################################

def train_neural_network(modelo,hidden_layer_sizes,neurons_per_layer,activation_function):
    (X_train,y_train) = modelo.get_training_data()
    if modelo.is_regresion():
        model = keras.models.Sequential()
        for i in range(hidden_layer_sizes):
            model.add(keras.layers.Dense(neurons_per_layer,activation=activation_function))
        model.add(keras.layers.Dense(1,activation="linear"))
        
        model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_squared_error"])
        model.fit(X_train,y_train,epochs=40)
    else:
        model = keras.models.Sequential()
        for i in range(hidden_layer_sizes):
            model.add(keras.layers.Dense(neurons_per_layer,activation=activation_function))
        model.add(keras.layers.Dense(y_train.shape[1],activation="softmax"))
        
        model.compile(loss="binary_crossentropy",     
                optimizer=keras.optimizers.Nadam(),
                metrics=["accuracy"])
        model.fit(X_train,y_train,epochs=40)
    
    return model

def train_neural_network_optuna(modelo):
    (X_train,y_train) = modelo.get_training_data()
    def objective(trial):
        seed(1)
        tf.random.set_seed(2)
        
        if modelo.is_regresion():
            
            hidden_layer_sizes = trial.suggest_int("hidden_layer_sizes",3,10)
            neurons_per_layer = trial.suggest_int("neurons_per_layer",60,200)
            activation_function = trial.suggest_categorical("activation_function", ['relu', 'selu',"softmax","tanh"])
            
            early_stopping_cb = keras.callbacks.EarlyStopping(patience=5,restore_best_weights=True) # Early Stopping
            
            model = keras.models.Sequential()
            for i in range(hidden_layer_sizes):
                model.add(keras.layers.Dense(neurons_per_layer,activation=activation_function))
            model.add(keras.layers.Dense(1))

            X_train_2, X_val, y_train_2, y_val = train_test_split(X_train, y_train, test_size=0.2)

            seed(1)
            tf.random.set_seed(2)
            model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_squared_error"],)
            
            model.fit(X_train_2,y_train_2,epochs=10,validation_data=(X_val,y_val),callbacks=[early_stopping_cb])
            score = -float(mean_squared_error(y_true=y_val,y_pred= model.predict(X_val)))
            return score
        else:
            hidden_layer_sizes = trial.suggest_int("hidden_layer_sizes",3,10)
            neurons_per_layer = trial.suggest_int("neurons_per_layer",60,200)
            activation_function = trial.suggest_categorical("activation_function", ['relu', 'selu',"softmax","tanh"])
            
            early_stopping_cb = keras.callbacks.EarlyStopping(patience=5,restore_best_weights=True) # Early Stopping
            
            model = keras.models.Sequential()
            for i in range(hidden_layer_sizes):
                model.add(keras.layers.Dense(neurons_per_layer,activation=activation_function))
            model.add(keras.layers.Dense(y_train.shape[1],activation="softmax"))

            X_train_2, X_val, y_train_2, y_val = train_test_split(X_train, y_train, test_size=0.2)
            
            seed(1)
            tf.random.set_seed(2)
            model.compile(loss="binary_crossentropy",     
                optimizer=keras.optimizers.Nadam(),
                metrics=["accuracy"])
            
            model.fit(X_train_2,y_train_2,epochs=10,validation_data=(X_val,y_val),callbacks=[early_stopping_cb])
            score = (balanced_accuracy_score(y_true=y_val.argmax(axis=1),y_pred= model.predict(X_val).argmax(axis=1)))
            return score
    
    sampler = TPESampler(seed=0) # create a seed for the sampler for reproducibility
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=18,n_jobs=3)
    
       
    seed(1)
    tf.random.set_seed(2)
    
    if modelo.is_regresion():
        early_stopping_cb = keras.callbacks.EarlyStopping(patience=5,restore_best_weights=True) # Early Stopping
        model = keras.models.Sequential()
        for i in range(study.best_params["hidden_layer_sizes"]):
            model.add(keras.layers.Dense(study.best_params["neurons_per_layer"],activation=study.best_params["activation_function"]))
        model.add(keras.layers.Dense(1))

        X_train_2, X_val, y_train_2, y_val = train_test_split(X_train, y_train, test_size=0.2)

        seed(1)
        tf.random.set_seed(2)
        model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_squared_error"],)
        
        model.fit(X_train_2,y_train_2,epochs=20,validation_data=(X_val,y_val),callbacks=[early_stopping_cb])
    else:
        early_stopping_cb = keras.callbacks.EarlyStopping(patience=5,restore_best_weights=True) # Early Stopping
        model = keras.models.Sequential()
        for i in range(study.best_params["hidden_layer_sizes"]):
            model.add(keras.layers.Dense(study.best_params["neurons_per_layer"],activation=study.best_params["activation_function"]))
        model.add(keras.layers.Dense(y_train.shape[1],activation="softmax"))

        X_train_2, X_val, y_train_2, y_val = train_test_split(X_train, y_train, test_size=0.2)
        
        seed(1)
        tf.random.set_seed(2)
        model.compile(loss="binary_crossentropy",     
                optimizer=keras.optimizers.Nadam(),
                metrics=["accuracy"])
            
        model.fit(X_train_2,y_train_2,epochs=40,validation_data=(X_val,y_val),callbacks=[early_stopping_cb])
    
    print("Se ha entrenado otra vez")
    #print(model)
    
    return (model,study)
    