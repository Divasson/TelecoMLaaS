from django.shortcuts import render,HttpResponse, redirect
import numpy as np
import pandas as pd
from AnalizarDatos.utils import utils
from CrearModelos.models import ModelosMachineLearning
from CrearAbrirProyectos.models import Project
from django.http import HttpResponse
from .forms import *

# Create your views here.
def subirPrediccion(request,id_project):
    project= utils.isUserLoggedIn_or_hisProject(request,id_project)
    if type(project)!=type(Project()):
        return redirect('/projects/')
    modelos_entrenados = project.get_list_modelos_entrenados()
    
    if len(modelos_entrenados)==0:
        return redirect('/modelsProject/'+str(id_project))
    
    context = {}
    context["proyecto"]= project
    context["todos_modelos_entrenados"] = project.is_all_models_trained()
    
    dict_modelos_entrenados = {}
    for modelo_entrenado_nombre in modelos_entrenados:
        modelo_clase = ModelosMachineLearning.objects.filter(proyecto=project,name=modelo_entrenado_nombre).first()
        dict_modelos_entrenados[modelo_entrenado_nombre] = modelo_clase.get_metrica_modelo()
        
    if project.is_regresion():
        dict_modelos_entrenados = dict(sorted(dict_modelos_entrenados.items(),key=lambda x:x[1],reverse=False))
    else:
        dict_modelos_entrenados = dict(sorted(dict_modelos_entrenados.items(),key=lambda x:x[1],reverse=True))
    
    
    choices = [(col,str(col)+"-"+str(round(dict_modelos_entrenados[col],2))) for col in list(dict_modelos_entrenados.keys())]
    context["formulario"] = FormularioModeloPredecir(choices=choices) 
    context["tipo"]=project.get_tipo_prediccion()
    
    
    if request.method == 'POST':
        form = FormularioModeloPredecir(request.POST, request.FILES, choices=choices)
        if form.is_valid():
            nombreModeloSeleccionado = form.cleaned_data["modeloPredecir"]
            archivo_a_predecir = request.FILES['archivo_a_predecir']
            file_type = archivo_a_predecir.content_type.split('/')[1]
            if file_type == 'csv':
                data = pd.read_csv(archivo_a_predecir)
            elif file_type == 'vnd.ms-excel' or file_type == 'vnd.openxmlformats-officedocument.spreadsheetml.sheet':
                data = pd.read_excel(archivo_a_predecir)
            else:
                return render(request, 'error.html', {'error_message': 'Invalid file type'})
        
            # hacer predicción
            data_transformed = project.transform_data(data)
            try:
                data_transformed = data_transformed.toarray()
            except:
                pass
            
            df_original = project.sacar_dataframe_original()
        
            modeloML = ModelosMachineLearning.objects.filter(proyecto=project,name=nombreModeloSeleccionado).first()
            prediccion = modeloML.predict(data_transformed)

            #df_original[project.get_variable_a_predecir()+"_predicted"] = np.nan
            df_original[project.get_variable_a_predecir()+"_predicted"] = "Faltan datos"
            
            df_original.loc[~df_original['is_deleted_row_in_na_process'], project.get_variable_a_predecir()+"_predicted"] = prediccion

            df_return = df_original.copy()
            df_return.drop(columns=["is_deleted_row_in_na_process"],inplace=True)
            
            # create a response object
            response = HttpResponse(content_type='application/ms-excel')
            response['Content-Disposition'] = 'attachment; filename="prediccion.xlsx"'

            # write the DataFrame to the response object as an Excel file
            df_return.to_excel(response, index=False)

            return response  
        

    return render(request, 'hacerPrediccion.html',context=context)
    
    
    

""" def hacerPrediccion(request,id_project):
    project= utils.isUserLoggedIn_or_hisProject(request,id_project)
    if type(project)!=type(Project()):
        return redirect('/projects/')
    modelos_entrenados = list(project.get_list_modelos_entrenados())
    if len(modelos_entrenados)==0:
        return redirect('/modelsProject/'+str(id_project))
    
    
    context = {}
    context["proyecto"]= project
    context["todos_modelos_entrenados"] = project.is_all_models_trained()
    
    
    
    dict_modelos_entrenados = {}
    for modelo_entrenado_nombre in modelos_entrenados:
        modelo_clase = ModelosMachineLearning.objects.filter(proyecto=project,name=modelo_entrenado_nombre).first()
        dict_modelos_entrenados[modelo_entrenado_nombre] = modelo_clase.get_metrica_modelo()
        
    if project.is_regresion():
        dict_modelos_entrenados = dict(sorted(dict_modelos_entrenados.items(),key=lambda x:x[1],reverse=False))
    else:
        dict_modelos_entrenados = dict(sorted(dict_modelos_entrenados.items(),key=lambda x:x[1],reverse=True))
    
    
    choices = [(col,str(col)+"-"+str(round(dict_modelos_entrenados[col],2))) for col in list(dict_modelos_entrenados.keys())]
    context["formulario"] = FormularioModeloPredecir(choices=choices)
    
    if request.POST:
        nombreModeloSeleccionado = request.POST.getlist("modeloPredecir")[0]
        data_transformed = project.sacar_datos_pred_temporales()
        df_original = project.sacar_dataframe_original()
        
        modeloML = ModelosMachineLearning.objects.filter(proyecto=project,name=nombreModeloSeleccionado).first()
        prediccion = modeloML.predict(data_transformed)

        df_original[project.get_variable_a_predecir()+"_predicted"] = np.nan
        
        df_original.loc[~df_original['is_deleted_row_in_na_process'], project.get_variable_a_predecir()+"_predicted"] = prediccion

        df_return = df_original.copy()
        df_return.drop(columns=["is_deleted_row_in_na_process"],inplace=True)
        
        # create a response object
        response = HttpResponse(content_type='application/ms-excel')
        response['Content-Disposition'] = 'attachment; filename="prediccion.xlsx"'

        # write the DataFrame to the response object as an Excel file
        df_return.to_excel(response, index=False)

        return response  
    
    return render(request,"verPrediccion.html",context=context)


 """