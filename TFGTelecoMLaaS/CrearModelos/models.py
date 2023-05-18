import json
import os

import joblib
import keras
import numpy as np
import pandas as pd
from django.db import models
from sklearn.metrics import (balanced_accuracy_score,
                             mean_absolute_percentage_error,
                             mean_squared_error)

from CrearAbrirProyectos.models import Project

from .entrenarModelos import entrenarModelos as MlaaS_train
from .utils import pipeline

listaModelosNotOHE = ["svc",
                      "logistic regression",
                      "svr",
                      ]

# Create your models here.
class ModelosMachineLearning(models.Model):
    proyecto = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='Proyecto')
    modelo   = models.CharField(max_length=200,blank=False,null=False)
    name = models.CharField(max_length=200,null=False,blank=False)
    optuna = models.BooleanField(blank=False,null=False,default=False)
    supports_y_ohe = models.BooleanField(blank=False,null=False,default=False)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    
    metrica_valorar = models.FloatField(null=True,blank=True)
    
    ensemble_model = models.BooleanField(blank=False,null=False,default=False)
    ensemble_models_made = models.CharField(max_length=255,null=True,blank=True)
    
    
    class Meta:
        unique_together = (('proyecto','name'),) 
    def __str__(self):
        return self.name
    
    ######################################################################################################
    ##########################################===IS===####################################################
    ######################################################################################################
    
    def is_optuna(self):
        return self.optuna
    
    def is_binary_model(self):
        return self.proyecto.is_binary_model()
    
    def is_regresion(self):
        return self.proyecto.is_regresion()
    
    def is_ensemble_model(self):
        return self.ensemble_model
    
    ######################################################################################################
    ##########################################===GETS===##################################################
    ######################################################################################################
    
    def get_metrica_modelo(self):
        return self.metrica_valorar
    def get_proyecto(self):
        return self.proyecto
    def get_nombre_modelo(self):
        return self.name
    def get_modelo(self):
        if self.name.lower() == "Neural Network".lower():
            return keras.models.load_model(self.modelo)
        else:
            return joblib.load(self.modelo)
    def get_study_optuna(self):
        return joblib.load("documents/models/project_{}/study_optuna_{}.pkl".format(self.proyecto.id,self.name))
    def get_supports_y_ohe(self):
        return self.supports_y_ohe
    def get_modelos_ensemble(self):
        lista_modelos_ensemble = list(json.loads(self.ensemble_models_made))
        modelos_ensemble = []
        for nombre_modelo in lista_modelos_ensemble:
            modelos_ensemble.append(ModelosMachineLearning.objects.filter(proyecto=self.proyecto,name=nombre_modelo).first())
        return modelos_ensemble
    
    def get_model_params(self):
        dict_response = {}
        if str(self.name).lower() == "Neural Network".lower():
            model = self.get_modelo()
            num_layers = len(model.layers)
            neurons = []
            activations = []
            for layer in model.layers:
                neurons.append(layer.output_shape[1])
                activations.append(layer.activation.__name__)
            dict_response["num_layers"] = num_layers
            dict_response["neurons"] = neurons
            dict_response["activations"] = activations
        else:
            params = self.get_modelo().get_params()
            if str(self.name).lower() == "knn":
                dict_response["n_neighbours"] = params["n_neighbors"]
                dict_response["weights"] = params["weights"]
                dict_response["metric"] = params["metric"]
            elif str(self.name).lower() == "svc":
                dict_response["kernel"] = params["kernel"]
                dict_response["C"] = params["C"]
                dict_response["gamma"] = params["gamma"]
            elif str(self.name).lower() == "svr":
                dict_response["kernel"] = params["kernel"]
                dict_response["C"] = params["C"]
                dict_response["gamma"] = params["gamma"]
                dict_response["degree"] = params["degree"]
            elif str(self.name).lower() == "Logistic Regression".lower():
                dict_response["C"] = params["C"]
                dict_response["solver"] = params["solver"]
                dict_response["penalty"] = params["penalty"]
            elif str(self.name).lower() == "Random Forest".lower():
                dict_response["criterion"] = params["criterion"]
                dict_response["n_estimators"] = params["n_estimators"]
                dict_response["max_depth"] = params["max_depth"]
                dict_response["min_samples_leaf"] = params["min_samples_leaf"]
            elif str(self.name).lower() == "ElasticNetCV".lower():
                dict_response["n_alphas"] = params["n_alphas"]
                dict_response["eps"] = params["eps"]
                dict_response["l1_ratio"] = params["l1_ratio"]
            else:
                dict_response = False
                
        return dict_response
    
    ######################################################################################################
    ########################################===PREDICT===#################################################
    ######################################################################################################
    
    def predict(self,X):
        
        if not self.is_ensemble_model():
            pred = self.get_modelo().predict(X)
            if not self.proyecto.is_regresion():
                if self.get_supports_y_ohe():
                    pred = pred.argmax(axis=1)    
            else:
                if self.get_nombre_modelo().lower() == "Neural Network".lower():
                    pred = np.concatenate(pred)
            return pred
        else:
            X_transformed = X.copy()
            modelos_ensemble = self.get_modelos_ensemble()
            
            if self.is_regresion():
                outcome = np.zeros(X_transformed.shape[0],)
                suma_inv_metricas = 0
                
                for modelo in modelos_ensemble:
                    
                    prediccion = modelo.predict(X_transformed)
                    metrica = (1/(modelo.get_metrica_modelo()))
                    suma_inv_metricas+=metrica
                    prediccion_ponderada = np.multiply(prediccion,metrica)
                    outcome+=prediccion_ponderada
                
                outcome = np.divide(outcome,suma_inv_metricas)
                
            else:
                outcome = np.zeros((X_transformed.shape[0],len(self.proyecto.get_prediction_dict().keys())))
                for modelo in modelos_ensemble:
                    prediccion = modelo.predict(X_transformed)
                    
                    prediccion_ohe = np.zeros((X_transformed.shape[0],len(self.proyecto.get_prediction_dict().keys())))
                    prediccion_ohe[np.arange(len(prediccion)),prediccion] = 1
                    
                    pred_ponderada = np.multiply(prediccion_ohe,modelo.get_metrica_modelo())
                    
                    outcome+=pred_ponderada
                
                for i in range(outcome.shape[0]):
                    max_index = np.argmax(outcome[i])
                    outcome[i] = 0
                    outcome[i][max_index] = 1
                    
                outcome = outcome.argmax(axis=1)
            return outcome
            
    
    def predict_proba(self,X):
        if not self.is_ensemble_model():
            if str(self.name).lower() == "Neural Network".lower():
                pred_proba = self.get_modelo().predict(X)
                return pred_proba
            else:
                pred_proba = np.array(self.get_modelo().predict_proba(X))
                if (self.name.lower() == "knn") | (self.name.lower()== "random forest"):
                    pred_proba = pred_proba[:,:,1].T
                return pred_proba
        else:
            modelos_ensemble = self.get_modelos_ensemble()
            total_pesos = 0
            proba_ponderada = 0
            for i,modelo in enumerate(modelos_ensemble):
                total_pesos+=modelo.get_metrica_modelo()
                predict_proba_modelo = modelo.predict_proba(X)
                if i==0:
                    proba_ponderada = np.multiply(predict_proba_modelo,modelo.get_metrica_modelo())
                else:
                    proba_ponderada += np.multiply(predict_proba_modelo,modelo.get_metrica_modelo())
            proba_ponderada = np.divide(proba_ponderada,total_pesos)
            
            return proba_ponderada
            
                
    
    ######################################################################################################
    ################################===GET TRAINING & TEST DATA===########################################
    ######################################################################################################
    
    def get_training_data(self):
        if self.is_regresion():
            return self.proyecto.get_training_data_Y_ohe()
        else:
            if self.get_supports_y_ohe():
                return self.proyecto.get_training_data_Y_ohe()
            else:
                return self.proyecto.get_training_data_Y_not_ohe()
        
    def get_test_data(self):
        if self.get_supports_y_ohe():
            return self.proyecto.get_raw_and_transf_test_data_ohe()
        else:
            return self.proyecto.get_raw_and_transf_test_data_Y_not_ohe()
    
    ######################################################################################################
    ##########################################===SETS===##################################################
    ######################################################################################################
    
    def set_proyecto(self,proyectoAsociado):
        self.proyecto=proyectoAsociado   
    def set_nombre_modelo(self,nombreModelo):
        self.name=nombreModelo
    def set_supports_y_ohe(self):
        if self.name.lower() not in listaModelosNotOHE:
            self.supports_y_ohe = True
        else:
            self.supports_y_ohe = False

    ######################################################################################################
    ########################################===ENSEMBLE===##################################################
    ######################################################################################################
    
    def rellenarModeloEnsemble(self,proyecto,nombres_modelos_ml_entrenados):
        nombre = "Ensemble de " + '-'.join(nombres_modelos_ml_entrenados)
        existing_mlmodel = ModelosMachineLearning.objects.filter(proyecto=proyecto,name=nombre).first()
        if existing_mlmodel:
            existing_mlmodel.delete()
            
        self.ensemble_models_made = json.dumps(nombres_modelos_ml_entrenados)
        self.proyecto = proyecto
        self.name= nombre
        self.supports_y_ohe = True
        self.optuna = False
        self.ensemble_model = True
        self.save_balanced_score()
        
    ######################################################################################################
    ########################################===OPTUNA===##################################################
    ######################################################################################################
    def rellenarModeloOptuna(self,proyecto,nombre_modelo_ml):
        try:
            existing_mlmodel = ModelosMachineLearning.objects.filter(proyecto=proyecto,name=nombre_modelo_ml).first()
            if existing_mlmodel:
                existing_mlmodel.delete()
        except:
            pass
        
        self.proyecto = proyecto
        self.name= nombre_modelo_ml
        self.set_supports_y_ohe()
        self.ensemble_model = False
        try:
            (modelo,study) = self.entrenarModeloOptuna(nombremodelo=nombre_modelo_ml)
            self.save_modelo(modelo=modelo)
            self.optuna = True
            self.save_optuna(study=study)
        except:
            (modelo) = self.entrenarModeloOptuna(nombremodelo=nombre_modelo_ml)
            self.save_modelo(modelo=modelo)
            self.optuna = False
        self.save_balanced_score()
        if (self.get_metrica_modelo()==0.5) & (not self.is_regresion()) & (str(self.name).lower()=="random forest"):
            raise("Este modelo no se ha entrenado bien")
        

    def entrenarModeloOptuna(self,nombremodelo):     
        if self.is_regresion():
            #-----------------------------------------------------------------------------------------------------
            ##################################===REGRESIÓN===#####################################################
            #-----------------------------------------------------------------------------------------------------
            if str(nombremodelo).lower()==("ElasticNetCV".lower()):
                return MlaaS_train.train_ElasticNetCV_optuna(modelo=self)    
            elif str(nombremodelo).lower()==("Random Forest".lower()):
                return MlaaS_train.train_RandomForestReg_optuna(modelo=self)
            elif str(nombremodelo).lower()==("KNN".lower()):
                return MlaaS_train.train_knnReg_optuna(modelo=self)
            elif str(nombremodelo).lower()==("Neural Network".lower()):
                return MlaaS_train.train_neural_network_optuna(modelo=self)
            elif str(nombremodelo).lower()==("Linear Regression".lower()):
                return MlaaS_train.train_linear_regression(modelo=self)
            elif str(nombremodelo).lower()==("SVR".lower()):
                return MlaaS_train.train_SVR_optuna(modelo=self)
        else:
            #-----------------------------------------------------------------------------------------------------
            ##################################===CLASIFICACIÓN===#################################################
            #-----------------------------------------------------------------------------------------------------
            if str(nombremodelo).lower()=="knn":
                return MlaaS_train.train_knnClass_optuna(modelo=self)
            elif str(nombremodelo).lower()=="svc":
                return MlaaS_train.train_SVC_optuna(modelo=self)
            elif str(nombremodelo).lower() == "Logistic Regression".lower():
                return MlaaS_train.train_LogisticRegression_optuna(modelo=self)
            elif str(nombremodelo).lower()==("Random Forest".lower()):
                return MlaaS_train.train_RandomForestClass_optuna(modelo=self)
            elif str(nombremodelo).lower()==("Neural Network".lower()):
                return MlaaS_train.train_neural_network_optuna(modelo=self)
        
        
        return False
    
    ######################################################################################################
    #######################################===PARÁMETROS===###############################################
    ######################################################################################################
    def rellenarModeloConParams(self,proyecto,nombre_modelo_ml,params):
        existing_mlmodel = ModelosMachineLearning.objects.filter(proyecto=proyecto,name=nombre_modelo_ml).first()
        if existing_mlmodel:
            existing_mlmodel.delete()
            
        self.proyecto = proyecto
        self.name= nombre_modelo_ml
        self.ensemble_model = False
        self.set_supports_y_ohe()
        modelo = self.entrenarModeloConParams(nombremodelo=nombre_modelo_ml,
                                                diccionarioParams=params)
        self.save_modelo(modelo=modelo,
                         id_proyecto=self.proyecto.id,
                         nombreModelo=self.name)
        
        self.save_balanced_score()
        
    def entrenarModeloConParams(self,nombremodelo,diccionarioParams):
        if self.is_regresion():
            #-----------------------------------------------------------------------------------------------------
            ##################################===REGRESIÓN===#####################################################
            #-----------------------------------------------------------------------------------------------------
            if str(nombremodelo).lower()==("ElasticNetCV".lower()):
                return MlaaS_train.train_ElasticNetCV(modelo=self,
                                                      l1_ratio=diccionarioParams["l1_ratio"],
                                                      eps=diccionarioParams["eps"],
                                                      n_alphas=diccionarioParams["n_alphas"])
            elif str(nombremodelo).lower()==("Random Forest".lower()):
                return MlaaS_train.train_RandomForestReg(modelo=self,
                                                        n_estimators=diccionarioParams["n_estimators"],
                                                        criterion=diccionarioParams["criterion"],
                                                        max_depth=diccionarioParams["max_depth"],
                                                        min_samples_leaf=diccionarioParams["min_samples_leaf"])
            elif str(nombremodelo).lower()=="KNN".lower():        
                return MlaaS_train.train_knnClass(modelo=self,
                                                    n_neighbors=diccionarioParams["n_neighbours"],
                                                    weights=diccionarioParams["weights"],
                                                    p= diccionarioParams["metric"])
            elif str(nombremodelo).lower()=="Neural Network".lower():
                return MlaaS_train.train_neural_network(modelo=self,
                                                        hidden_layer_sizes=diccionarioParams["hidden_layer_sizes"],
                                                        neurons_per_layer=diccionarioParams["neurons_per_layer"],
                                                        activation_function=diccionarioParams["activation_function"])
            elif str(nombremodelo).lower()=="SVR".lower():
                return MlaaS_train.train_SVR(modelo=self,
                                            C=diccionarioParams["C"],
                                            gamma=diccionarioParams["gamma"],
                                            kernel=diccionarioParams["kernel"],
                                            degree=diccionarioParams["degree"])
            elif str(nombremodelo).lower() == "Linear Regression".lower():
                return MlaaS_train.train_linear_regression(modelo=self)
            
        else:
            #-----------------------------------------------------------------------------------------------------
            ##################################===CLASIFICACIÓN===#################################################
            #-----------------------------------------------------------------------------------------------------
            if str(nombremodelo).lower()=="KNN".lower():        
                return MlaaS_train.train_knnClass(modelo=self,
                                                n_neighbors=diccionarioParams["n_neighbours"],
                                                weights=diccionarioParams["weights"],
                                                p= diccionarioParams["metric"])
            elif str(nombremodelo).lower()=="SVC".lower():
                return MlaaS_train.train_SVC(modelo=self,
                                            C=diccionarioParams["C"],
                                            gamma=diccionarioParams["gamma"],
                                            kernel=diccionarioParams["kernel"])
            elif str(nombremodelo).lower()=="Logistic Regression".lower():
                return MlaaS_train.train_LogisticRegression(modelo=self,
                                                            C=diccionarioParams["C"],
                                                            solver=diccionarioParams["solver"],
                                                            penalty=diccionarioParams["penalty"])
            elif str(nombremodelo).lower()==("Random Forest".lower()):
                return MlaaS_train.train_RandomForestClass(modelo=self,
                                                        n_estimators=diccionarioParams["n_estimators"],
                                                        criterion=diccionarioParams["criterion"],
                                                        max_depth=diccionarioParams["max_depth"],
                                                        min_samples_leaf=diccionarioParams["min_samples_leaf"])
            elif str(nombremodelo).lower()=="Neural Network".lower():
                return MlaaS_train.train_neural_network(modelo=self,
                                                        hidden_layer_sizes=diccionarioParams["hidden_layer_sizes"],
                                                        neurons_per_layer=diccionarioParams["neurons_per_layer"],
                                                        activation_function=diccionarioParams["activation_function"])
            
        return False
    
    ######################################################################################################
    ###########################################===SAVE===#################################################
    ######################################################################################################
        
    
    def save_modelo(self,modelo):
        filename = 'documents/models/project_{}/{}.h5'.format(self.get_proyecto().id,self.name.lower())
        try:
            os.mkdir('documents/models/project_{}'.format(self.get_proyecto().id))
        except:
            pass
        if self.name.lower() == "Neural Network".lower():
            modelo.save(filename)
        else:
            joblib.dump(modelo,filename)
        self.modelo = filename
        
    def save_optuna(self,study):
        joblib.dump(study,"documents/models/project_{}/study_optuna_{}.pkl".format(self.proyecto.id,self.name))

    def save_balanced_score(self):
        (_,X_test_transformed,y_test_ohe) = self.proyecto.get_raw_and_transf_test_data_ohe()
        if self.proyecto.is_regresion():
            self.metrica_valorar = mean_squared_error(y_true=y_test_ohe,
                                                        y_pred=self.predict(X_test_transformed),squared=False)
        else:
            try:
                self.metrica_valorar = balanced_accuracy_score(y_true=y_test_ohe.argmax(axis=1),y_pred=self.predict(X_test_transformed).argmax(axis=1))
            except:
                self.metrica_valorar = balanced_accuracy_score(y_true=y_test_ohe.argmax(axis=1),y_pred=self.predict(X_test_transformed))
            
                
        print("metrica: "+str(self.metrica_valorar))
            
        
