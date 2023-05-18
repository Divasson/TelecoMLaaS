import errno
import shutil
from django.db import models
from django.db import models
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import CrearModelos
from Autenticacion.models import Usuario
from django.conf import settings
import pandas as pd
import json
from CrearModelos.utils.pipeline import X_columnTransformer_OHE
import numpy as np
import joblib
from datetime import datetime
import os


# Create your models here.
class ProjectManager(models.Manager):
    pass


class Project(models.Model):
    usuario = models.ForeignKey(Usuario, on_delete=models.CASCADE, related_name='projects')
    projectName = models.CharField(max_length=100)
    archivoDatos = models.FileField(upload_to='documents/datasetsProyectos/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    project_state = models.IntegerField(default=0)
    
    variable_a_predecir = models.CharField(max_length=100,editable=True,blank=True)

    tipo_prediccion = models.CharField(max_length=50,choices=[("clasificacion","clasificacion"),("regresion","regresion")],blank=True)
    tiposDatosProcesados = models.FileField(upload_to="documents/tiposDatosProcesados/",blank=True)
    
    archivoDatosNA= models.FileField(upload_to="documents/datasetsProyectosProcesados/",blank=True)
    
    version_datos_a_usar = models.DateTimeField(auto_now_add=True)
    tipos_de_modelos = models.CharField(max_length=255,null=True,blank=True)
    
    # train_test_split 
    train_test_split = models.FloatField(max_length=100,blank=True,null=True)
    test_val_split = models.FloatField(max_length=100,blank=True,null=True)
    
    # se han preprocesado los datos?
    preprocesado = models.BooleanField(default=0)
    
    treat_na_dict = models.FileField(upload_to="documents/treat_na/",blank=True)
    
    columnas_originales = models.CharField(max_length=5000,null=True,blank=True)
    
    
    class Meta:
        unique_together = (('usuario','projectName'),)
       

    def __str__(self):
        return self.projectName
    
    ######################################################################################################
    ######################################## ===  IS  === ################################################
    ######################################################################################################
    
    def is_archivo_datos_without_na(self):
        if self.archivoDatosNA:
            return True
        return False
    
    def is_archivo_dtypes(self):
        if self.tiposDatosProcesados:
            return True
        return False
    
    def is_archivo_tratar_na(self):
        if self.treat_na_dict:
            return True
        return False
    
    def is_regresion(self):
        if self.tipo_prediccion=="regresion":
            return True
        return False
    def is_preprocesado(self):
        return self.preprocesado
    
    def is_train_test_split_done(self):
        return self.preprocesado

    def is_modelo_entrenado(self,nombre_modelo_ml):
        if CrearModelos.models.ModelosMachineLearning.objects.filter(proyecto=self,name=nombre_modelo_ml).first():
            return True
        return False
    
    def is_modelos_seleccionados(self):
        try:
            a = self.get_modelos_seleccionados()
            return True
        except:
            return False
        return False
    
    def is_all_models_trained(self):
        modelos_a_entrenar = self.get_modelos_seleccionados()
        modelos_entrenados = self.get_list_modelos_entrenados()
        
        if len(modelos_a_entrenar)<=len(modelos_entrenados):
            return True
        return False
    
    def is_binary_model(self):
        if len(self.get_data()[self.get_variable_a_predecir()].unique())<3:
            return True
        return False
    
    ######################################################################################################
    ####################################### ===  SET  === ################################################
    ######################################################################################################
    
    def set_modelos_seleccionados(self,x):
        self.tipos_de_modelos = json.dumps(x)

    def set_train_test_val_splits(self,train_test_split,test_val_split):
        self.train_test_split = train_test_split/100
        self.test_val_split = test_val_split/100
    
    def set_columnas_originales(self,x):
        self.columnas_originales = json.dumps(x)
    
    ######################################################################################################
    ####################################### ===  DELETE  === ################################################
    ######################################################################################################
    
    def delete_modelos_asociados(self):
        lista_modelos_asociados = CrearModelos.models.ModelosMachineLearning.objects.filter(proyecto=self)
        if len(lista_modelos_asociados)>0:
            for modelo_entrenado in lista_modelos_asociados:
                modelo_entrenado.delete()
            if os.path.exists(str(settings.BASE_DIR)+"/documents/models/project_{}".format(self.id)):
                shutil.rmtree(str(settings.BASE_DIR)+"/documents/models/project_{}".format(self.id))
        return True
    
    def delete_treat_na_files(self):
        if self.is_archivo_tratar_na():
            for file in os.listdir(str(settings.BASE_DIR)+"/documents/treat_na/"):
                if file.startswith("project_{}_d".format(self.id)):
                    print(file)
                    os.remove(str(settings.BASE_DIR)+"/documents/treat_na/"+file)
        return True
    
    def delete_temp_files(self):
        if os.path.exists(str(settings.BASE_DIR)+"/documents/temp_files/{}".format(self.id)):
            shutil.rmtree(str(settings.BASE_DIR)+"/documents/temp_files/{}".format(self.id))
        return True
    
    def delete_dtypes_files_with_project(self):
        if self.is_archivo_dtypes():
            for file in os.listdir(str(settings.BASE_DIR)+"/documents/tiposDatosProcesados/"):
                if file.startswith("project_{}_d".format(self.id)):
                    print(file)
                    os.remove(str(settings.BASE_DIR)+"/documents/tiposDatosProcesados/"+file)
        return True
    
    def delete_after_pipe_files(self):
        if os.path.isdir(str(settings.BASE_DIR)+"/documents/data_after_pipe/"+str(self.id)):
            for root, dirs, files in os.walk(str(settings.BASE_DIR)+"/documents/data_after_pipe/"+str(self.id), topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(str(settings.BASE_DIR)+"/documents/data_after_pipe/"+str(self.id))
    
    def delete_datasets_processed(self):
        if self.is_archivo_datos_without_na():
            for file in os.listdir(str(settings.BASE_DIR)+"/documents/datasetsProyectosProcesados/"):
                if file.startswith("project_{}_d".format(self.id)):
                    print(file)
                    os.remove(str(settings.BASE_DIR)+"/documents/datasetsProyectosProcesados/"+file)
        return True
        
    
    def borrar_todos_archivos_vinculados(self):
        #borrado de modelos asociados
        self.delete_modelos_asociados()
        # borro archivos de tratar na
        self.delete_treat_na_files()
        # borro archivos temporales de prediccion
        self.delete_temp_files()
        # borro archivos tipos de datos
        self.delete_dtypes_files_with_project()
        # borro datos after_pipe
        self.delete_after_pipe_files()
        # borro datos proyectos procesados
        self.delete_datasets_processed()
        # borro fichero asociado al proyecto
        os.remove(str(settings.BASE_DIR)+str(self.archivoDatos.url))
        
        return True
        
        
    
    ######################################################################################################
    ####################################### ===  GET  === ################################################
    ######################################################################################################
    
    #-----------------------------------------------------------------------------------------------------
    #------------------------------------===GET ATTRIBUTES===---------------------------------------------
    #-----------------------------------------------------------------------------------------------------
    
    def get_tipo_prediccion(self):
        return self.tipo_prediccion
    
    def get_nombre_proyecto(self):
        return self.projectName
    
    def get_variable_a_predecir(self):
        return self.variable_a_predecir
    
    def get_version_datos(self):
        return (str(self.version_datos_a_usar.day) + "/" +str(self.version_datos_a_usar.month)+"/"+str(self.version_datos_a_usar.year)+"-"+str(self.version_datos_a_usar.hour)+":"+str(self.version_datos_a_usar.minute) )
    
    def get_modelos_seleccionados(self):
        return list(json.loads(self.tipos_de_modelos))
    
    def get_columnas_originales(self):
        return list(json.loads(self.columnas_originales))
    
    def get_list_modelos_entrenados(self):
        listaModelosEntrenados = CrearModelos.models.ModelosMachineLearning.objects.filter(proyecto=self)
        ret = []
        for modelo in listaModelosEntrenados:
            ret.append(modelo.get_nombre_modelo())
        return ret
    
    def get_all_columns_data(self):
        return self.get_data().columns
    
    def get_columns_X(self):
        return np.setdiff1d(self.get_data().columns,self.variable_a_predecir)
    
    def get_raw_dtypes(self):
        try:
            with open(str(settings.BASE_DIR)+str(self.tiposDatosProcesados.url), 'r') as j:
                tipos = json.loads(j.read())
            return tipos
        except:
            raise Exception("No hay datos preprocesados")
        
    def get_tratar_na(self):
        with open(str(settings.BASE_DIR)+str(self.treat_na_dict.url), 'r') as j:
            tratar_na = dict(json.loads(j.read()))
            return tratar_na
    
    
    
    #-----------------------------------------------------------------------------------------------------
    #------------------------------------------===GET DATA===---------------------------------------------
    #-----------------------------------------------------------------------------------------------------

        
        
    def get_data(self, original = False):
        if original:
            if self.is_archivo_dtypes():
                with open(str(settings.BASE_DIR)+str(self.tiposDatosProcesados.url), 'r') as j:
                        tipos = dict(json.loads(j.read()))
                        keys = [k for k, v in dict(tipos).items() if 'datetime' in v]

                        if self.archivoDatos.url.endswith('.csv'):
                            if len(keys)>0:
                                for key in keys:
                                    tipos[key] = "str"
                                
                                return pd.read_csv(str(settings.BASE_DIR)+str(self.archivoDatos.url),dtype=tipos,index_col=False,parse_dates=keys)
                            else:
                                return pd.read_csv(str(settings.BASE_DIR)+str(self.archivoDatos.url),dtype=tipos,index_col=False)
                        else:
                            if len(keys)>0:
                                for key in keys:
                                    tipos[key] = "str"
                                return pd.read_excel(str(settings.BASE_DIR)+str(self.archivoDatos.url),dtype=tipos,index_col=False,parse_dates=keys)
                            else:
                                return pd.read_excel(str(settings.BASE_DIR)+str(self.archivoDatos.url),dtype=tipos,index_col=False)
                            
            elif self.archivoDatos.url.endswith('.csv'):
                return pd.read_csv(str(settings.BASE_DIR)+str(self.archivoDatos.url),index_col=False)
            else:
                return pd.read_excel(str(settings.BASE_DIR)+str(self.archivoDatos.url),index_col=False)
        else:
            if self.is_archivo_datos_without_na():
                if self.is_archivo_dtypes():
                    with open(str(settings.BASE_DIR)+str(self.tiposDatosProcesados.url), 'r') as j:
                        tipos = dict(json.loads(j.read()))
                        keys = [k for k, v in dict(tipos).items() if 'datetime' in v]
                        if self.archivoDatosNA.url.endswith('.csv'):
                            if len(keys)>0:
                                for key in keys:
                                    tipos[key] = "str"
                                return pd.read_csv(str(settings.BASE_DIR)+str(self.archivoDatosNA.url),dtype=tipos,index_col=False,parse_dates=keys)
                            else:
                                return pd.read_csv(str(settings.BASE_DIR)+str(self.archivoDatosNA.url),dtype=tipos,index_col=False)
                        else:
                            if len(keys)>0:
                                for key in keys:
                                    tipos[key] = "str"
                                return pd.read_excel(str(settings.BASE_DIR)+str(self.archivoDatosNA.url),dtype=tipos,index_col=False,parse_dates=keys)
                            else:
                                return pd.read_excel(str(settings.BASE_DIR)+str(self.archivoDatosNA.url),dtype=tipos,index_col=False)
            
            
            elif self.is_archivo_dtypes():
                with open(str(settings.BASE_DIR)+str(self.tiposDatosProcesados.url), 'r') as j:
                        tipos = dict(json.loads(j.read()))
                        keys = [k for k, v in dict(tipos).items() if 'datetime' in v]

                        if self.archivoDatos.url.endswith('.csv'):
                            if len(keys)>0:
                                for key in keys:
                                    tipos[key] = "str"
                                
                                return pd.read_csv(str(settings.BASE_DIR)+str(self.archivoDatos.url),dtype=tipos,index_col=False,parse_dates=keys)
                            else:
                                return pd.read_csv(str(settings.BASE_DIR)+str(self.archivoDatos.url),dtype=tipos,index_col=False)
                        else:
                            if len(keys)>0:
                                for key in keys:
                                    tipos[key] = "str"
                                return pd.read_excel(str(settings.BASE_DIR)+str(self.archivoDatos.url),dtype=tipos,index_col=False,parse_dates=keys)
                            else:
                                return pd.read_excel(str(settings.BASE_DIR)+str(self.archivoDatos.url),dtype=tipos,index_col=False)
        
            elif self.archivoDatos.url.endswith('.csv'):
                return pd.read_csv(str(settings.BASE_DIR)+str(self.archivoDatos.url),index_col=False)
            else:
                return pd.read_excel(str(settings.BASE_DIR)+str(self.archivoDatos.url),index_col=False)
    
    
    ######################################################################################################
    ############################### ===  SAVE PREPROCESSED DATA  === #####################################
    ######################################################################################################
    
    def preprocesar_datos_y_guardarlos(self):
        
        # si voy a guardar nuevos datos, estos pueden haber cambiado de los anteriores y hay que borrar los modelos
        self.delete_modelos_asociados()
        if self.is_regresion():
            (col_transformer,
            X_train_transformed, 
            X_test_transformed, 
            X_train, 
            X_test, 
            y_train_ohe,
            y_test_ohe,
            cols_X_transformed) = X_columnTransformer_OHE(df=self.get_data(),
                                                            target=self.get_variable_a_predecir(),
                                                            train_split=self.train_test_split,
                                                            is_regression=True) 
        else:
            df = self.get_data()
            
            df[self.get_variable_a_predecir()] = df[self.get_variable_a_predecir()].astype("object")
            
            
            (col_transformer,
            X_train_transformed, 
            X_test_transformed, 
            X_train, 
            X_test, 
            y_train_ohe,
            y_test_ohe,
            pred_dict,
            cols_X_transformed) = X_columnTransformer_OHE(df=df,
                                                            target=self.get_variable_a_predecir(),
                                                            train_split=self.train_test_split,
                                                            is_regression=False)    
            
        
        remove_previous_data_after_pipe(self.id)
        
        ##################################### ===  Transformed  === ############################################
        try:
            np.save("documents/data_after_pipe/"+str(self.id)+"/X_train_transformed.npy",X_train_transformed.toarray(),allow_pickle=True)
        except:
            np.save("documents/data_after_pipe/"+str(self.id)+"/X_train_transformed.npy",X_train_transformed,allow_pickle=True)
        try:
            np.save("documents/data_after_pipe/"+str(self.id)+"/X_test_transformed.npy",X_test_transformed.toarray(),allow_pickle=True)
        except:
            np.save("documents/data_after_pipe/"+str(self.id)+"/X_test_transformed.npy",X_test_transformed,allow_pickle=True)
            
            
        ######################################### ===  Raw  === #################################################    
        try:
            np.save("documents/data_after_pipe/"+str(self.id)+"/X_train.npy",X_train.toarray(),allow_pickle=True)
        except:
            np.save("documents/data_after_pipe/"+str(self.id)+"/X_train.npy",X_train,allow_pickle=True)
        try:
            np.save("documents/data_after_pipe/"+str(self.id)+"/X_test.npy",X_test.toarray(),allow_pickle=True)
        except:
            np.save("documents/data_after_pipe/"+str(self.id)+"/X_test.npy",X_test,allow_pickle=True)
            
        np.save("documents/data_after_pipe/"+str(self.id)+"/columnas_datos_guardados.npy",X_train.columns,allow_pickle=True)
        
        
        ######################################## ===  Label OHE  === #################################################
        try:
            np.save("documents/data_after_pipe/"+str(self.id)+"/y_train_ohe",y_train_ohe.toarray(),allow_pickle=True)
        except:
            np.save("documents/data_after_pipe/"+str(self.id)+"/y_train_ohe",y_train_ohe,allow_pickle=True)
        try:
            np.save("documents/data_after_pipe/"+str(self.id)+"/y_test_ohe",y_test_ohe.toarray(),allow_pickle=True)
        except:    
            np.save("documents/data_after_pipe/"+str(self.id)+"/y_test_ohe",y_test_ohe,allow_pickle=True)
        
        
        ###################################### ===  Prediciton Dictionary  === ########################################
        if not self.is_regresion():
            try:
                np.save("documents/data_after_pipe/"+str(self.id)+"/pred_dict",pred_dict.toarray(),allow_pickle=True)
            except:
                np.save("documents/data_after_pipe/"+str(self.id)+"/pred_dict",pred_dict,allow_pickle=True)
        else:
            pass
        
        
        ##################################### ===  Columnas Transformadas  === ########################################

        np.save("documents/data_after_pipe/"+str(self.id)+"/cols_X_trans",cols_X_transformed,allow_pickle=True)
        joblib.dump(col_transformer.get_params(),"documents/data_after_pipe/"+str(self.id)+"/col_transformer_params.pkl",compress=True)
        joblib.dump(col_transformer, "documents/data_after_pipe/"+str(self.id)+"/col_transformer.pkl")
        
        
        ##################################### ===  Actualizar atributos  === ########################################
        self.version_datos_a_usar = datetime.now()
        self.preprocesado = True
        
    ######################################################################################################
    ############################### ===  GET PREPROCESSED DATA  === ######################################
    ######################################################################################################
    
    #-----------------------------------------------------------------------------------------------------
    #---------------------------------===GET RANSFORMATION DATA===----------------------------------------
    #-----------------------------------------------------------------------------------------------------
    
    def get_columns_X_transformed(self):
        return list(np.load("documents/data_after_pipe/"+str(self.id)+"/cols_X_trans.npy",allow_pickle=True))
    
    def get_col_transformer(self):

        """ columns_originales =np.load("documents/data_after_pipe/"+str(self.id)+"/columnas_datos_guardados.npy",allow_pickle=True)
        X_train = np.load("documents/data_after_pipe/"+str(self.id)+"/X_train.npy",
                                       allow_pickle=True)

        X_train_df = pd.DataFrame(X_train,
                                    columns=columns_originales) """
        
        
        full_data = self.get_data()
        target = self.get_variable_a_predecir()
        INPUTS = np.setdiff1d(full_data.columns,target)
        OUTPUT = target
        
        X = pd.DataFrame(full_data[INPUTS])
        y = pd.DataFrame(full_data[OUTPUT])
        
        X_train, _, __, ___ = train_test_split(X, y,
                                                        test_size=self.train_test_split, 
                                                        random_state=0)        
        
        INPUTS_NUM = X.select_dtypes(include=['int64','float64']).columns.values.tolist()
        INPUTS_CAT = X.select_dtypes(include=['object']).columns.values.tolist()
        
        
        col_transformer = ColumnTransformer(transformers=[
                                        ('num', StandardScaler(), INPUTS_NUM),
                                        ('cat', OneHotEncoder(handle_unknown='ignore',drop="first"), INPUTS_CAT)
                                        ])

        col_transformer.fit(X_train)
        return (col_transformer)
        
    
    def get_prediction_dict(self):
        if not self.is_regresion():
            return dict(np.load("documents/data_after_pipe/"+str(self.id)+"/pred_dict.npy",allow_pickle=True).tolist())    
        else:
            raise Exception("No es clasificación")
    
    
    #-----------------------------------------------------------------------------------------------------
    #-----------------------------===FOR TRAINING CLASSIFICATION MODELS===--------------------------------
    #-----------------------------------------------------------------------------------------------------

    def get_training_data_Y_not_ohe(self):
        (X_train,y_train_ohe) = self.get_training_data_Y_ohe()
        return (X_train,y_train_ohe.argmax(axis=1))
    
    def get_training_data_Y_ohe(self):
        X_train = np.load("documents/data_after_pipe/"+str(self.id)+"/X_train_transformed.npy",allow_pickle=True)
        #X_val = np.load("documents/data_after_pipe/"+str(self.id)+"/X_val_transformed.npy",allow_pickle=True)
        y_train = np.load("documents/data_after_pipe/"+str(self.id)+"/y_train_ohe.npy",allow_pickle=True)
        #y_val = np.load("documents/data_after_pipe/"+str(self.id)+"/y_val_ohe.npy",allow_pickle=True)
        return (X_train,y_train)
    
    
    #-----------------------------------------------------------------------------------------------------
    #----------------------------------------===GET TEST DATA===------------------------------------------
    #-----------------------------------------------------------------------------------------------------
    def get_raw_test_data_ohe(self):
        X_test = np.load("documents/data_after_pipe/"+str(self.id)+"/X_test.npy",allow_pickle=True)
        y_test = np.load("documents/data_after_pipe/"+str(self.id)+"/y_test_ohe.npy",allow_pickle=True)
        return (X_test,y_test)
    
    def get_raw_and_transf_test_data_ohe(self):
        (X_test_raw,y_test) = self.get_raw_test_data_ohe()
        X_test_transformed = np.load("documents/data_after_pipe/"+str(self.id)+"/X_test_transformed.npy",allow_pickle=True)
        return (X_test_raw,X_test_transformed,y_test)
    
    def get_raw_and_transf_test_data_Y_not_ohe(self):
        (X_test_raw,X_test_transformed,y_test) = self.get_raw_and_transf_test_data_ohe()
        return (X_test_raw,X_test_transformed,y_test.argmax(axis=1))
    
    
    
    #-----------------------------------------------------------------------------------------------------
    #-------------------------------------===Transformar Datos===-----------------------------------------
    #-----------------------------------------------------------------------------------------------------

    def transform_data(self,X):
        
        cols_originales = self.get_columnas_originales()
        cols_originales.remove(self.get_variable_a_predecir())
        X_df = pd.DataFrame(X,columns=cols_originales)
        
        X_df_aux = X_df.copy()
        X_df_aux["is_deleted_row_in_na_process"] = False
        cols_treated = []
        if self.is_archivo_datos_without_na():
            tratar_na = self.get_tratar_na()
            for key in tratar_na.keys():
                cols_treated.append(key)
                
                if tratar_na[key]["tipo"]=="0":
                    X_df[key]=X_df[key].fillna("0")   
                                
                elif tratar_na[key]["tipo"]=="median":
                    X_df[key]=X_df[key].fillna(tratar_na[key]["valor"])
                
                elif tratar_na[key]["tipo"]=="del":
                    X_df_aux['is_deleted_row_in_na_process'] = X_df_aux[key].isna() | X_df_aux['is_deleted_row_in_na_process']
                    X_df.dropna(subset=[key],inplace=True)
                
                elif tratar_na[key]["tipo"]=="labelMostUsed":
                    X_df[key]=X_df[key].fillna(tratar_na[key]["valor"])
                
                elif tratar_na[key]["tipo"]=="delCol":
                    X_df.drop(columns=[key],inplace=True)        
        
        cols_to_check = np.setdiff1d(X_df_aux.columns,cols_treated)
        for index, row in X_df_aux.iterrows():
            for col in cols_to_check:
                if pd.isna(row[col]):
                    X_df_aux.at[index,'is_deleted_row_in_na_process'] = True
                    break  # exit the inner loop once a NaN value is found
        

        
        self.guardar_dataframe_original(X_df_aux)
        
        
        X_df.dropna(inplace=True)

        col_transformer = self.get_col_transformer()
        
        
        X_transformed = col_transformer.transform(X_df)
        
        
        
        return X_transformed
                    
    #-----------------------------------------------------------------------------------------------------
    #-------------------------------------===Guardar Datos Pred===-----------------------------------------
    #-----------------------------------------------------------------------------------------------------           

    def guardar_datos_pred_temporalmente(self,X):
        try:
            np.save("documents/temp_files/"+str(self.id)+"/temp_pred.npy",X.toarray(),allow_pickle=True)
        except:
            np.save("documents/temp_files/"+str(self.id)+"/temp_pred.npy",X,allow_pickle=True)
    def sacar_datos_pred_temporales(self):
        return np.load("documents/temp_files/"+str(self.id)+"/temp_pred.npy",allow_pickle=True)
        
    def guardar_dataframe_original(self,df):
        path = str(settings.BASE_DIR)+"/documents/temp_files/"+str(self.id)
        try:
            os.makedirs(path)
        except OSError as exc: # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else: raise
            
        df.to_excel(path+"/original.xlsx", index=False)
        
        
    def sacar_dataframe_original(self):
        path = str(settings.BASE_DIR)+"/documents/temp_files/"+str(self.id)
        if os.path.isdir(path):
            return pd.read_excel(path+"/original.xlsx")  
        else:
            return False
    

######################################################################################################
################################## ===  HELP FUNCTIONS  === ##########################################
######################################################################################################

def remove_previous_data_after_pipe(id_):
    if os.path.isdir("documents/data_after_pipe/"+str(id_)):
        for root, dirs, files in os.walk("documents/data_after_pipe/"+str(id_), topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir("documents/data_after_pipe/"+str(id_))
    os.mkdir("documents/data_after_pipe/"+str(id_))