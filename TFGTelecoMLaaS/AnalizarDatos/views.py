import json
from django.shortcuts import redirect, render
from CrearAbrirProyectos.models import Project
import pandas as pd
from AnalizarDatos import plotlyDescriptivo
from .forms import ChangeDataTypeForm, FormularioVariables,FormularioVariableaPredecir,FormularioTipoPrediccion, TratarNA
from AnalizarDatos.utils import utils
import numpy as np
from django.core.files.base import ContentFile
from sklearn.utils import shuffle
from datetime import datetime


    ######################################################################################################
    #################################===Seleccionar Variable===###########################################
    ######################################################################################################
def start_project(request,id_project,cambiarVar=None):
    project= utils.isUserLoggedIn_or_hisProject(request,id_project)
    if type(project)!=type(Project()):
        return redirect('/projects/')   
    
    if not cambiarVar:
        if project.project_state!=0:
            return redirect('/initProject/analisisDescriptivo/tipoPrediccion/'+str(id_project))
        
    context = {}
    context["proyecto"] = project
    
    if request.POST:
        variable1 = request.POST.getlist("variablePredecir")[0]
        project.variable_a_predecir = variable1
        project.project_state = 1
        project.save()
        return redirect('/initProject/analisisDescriptivo/tipoPrediccion/'+str(id_project))
    
    df = project.get_data()

    variables = df.columns.to_list()
    choices = [(col, col) for col in variables]
    if project.variable_a_predecir:
        formulario_var_a_predecir = FormularioVariableaPredecir(choices,initialVar=project.variable_a_predecir)
        context["formulario_variable"] = formulario_var_a_predecir
    else:
        formulario_var_a_predecir = FormularioVariableaPredecir(choices,initialVar=variables[0])
        context["formulario_variable"] = formulario_var_a_predecir
    return render(request,"projectChildren/analisisDescriptivo/seleccionarVariable.html",context=context)


    ######################################################################################################
    ###################################===Tipo Prediccion===##############################################
    ######################################################################################################
   
def confirmarTipoPrediccion(request,id_project,cambiarVar=None):
    project= utils.isUserLoggedIn_or_hisProject(request,id_project)
    if type(project)!=type(Project()):
        return redirect('/projects/')
    
    if not cambiarVar:
        if project.project_state<1:
            return redirect('/initProject/analisisDescriptivo/seleccionarVariable/'+str(id_project))
        elif project.project_state>1:
            return redirect('/initProject/analisisDescriptivo/confirmarDatos/'+str(id_project))
        #return redirect('/initProject/analisisDescriptivo/visualizarDatos/'+str(id_project))
    
    context = {}
    context["proyecto"] = project
    
    df = project.get_data()
    
    tipo_pred_regresion = 0
    if np.issubdtype(df[project.variable_a_predecir].dtype, np.number):
        df_aux =  df.copy()
        df_aux[project.get_variable_a_predecir()] = df_aux[project.get_variable_a_predecir()].astype("object")
        if df_aux[project.get_variable_a_predecir()].value_counts().iloc[0]<30:
            tipo_pred_regresion = 1
        

    if request.POST:
        form1 = 0
        if tipo_pred_regresion:
            form1 = FormularioTipoPrediccion(request.POST,type_of_prediction="regresion")
        else:
            form1 = FormularioTipoPrediccion(request.POST,type_of_prediction="clasificacion")
        if form1.is_valid():
            tipo = form1.cleaned_data['prediction_type']            
            if ((tipo_pred_regresion & (tipo=="regresion")) |
                ((tipo == "clasificacion"))):
                project.project_state = 2
                project.tipo_prediccion = tipo
                project.save()
                return redirect('/initProject/analisisDescriptivo/confirmarDatos/'+str(id_project))
                
            else:
                print("NOT VALID")    
                #return redirect('/initProject/analisisDescriptivo/confirmarDatos/'+str(id_project))
        else:
            print("NOT VALID FORM")

    if tipo_pred_regresion:
        context["form"] = FormularioTipoPrediccion(type_of_prediction="regresion")
        context["tipoPred"]= 0
    else:
        context["form"] = FormularioTipoPrediccion(type_of_prediction="clasificacion")
        context["tipoPred"]= 1
    return render(request,"projectChildren/analisisDescriptivo/confirmarTipoPrediccion.html",context=context)


    ######################################################################################################
    ###############################===Cambiar tipos de datos===###########################################
    ######################################################################################################


def confirm_data(request,id_project,cambiarVar=None):
    project= utils.isUserLoggedIn_or_hisProject(request,id_project)
    if type(project)!=type(Project()):
        return redirect('/projects/')
    
    if not cambiarVar:
        if project.project_state<2:
            return redirect('/initProject/analisisDescriptivo/tipoPrediccion/'+str(id_project))
        elif project.project_state>2:
            return redirect('/initProject/analisisDescriptivo/tratarNa/'+str(id_project))
    
    context = {}
    context["proyecto"] = project
    context["tipo"]=project.get_tipo_prediccion()
    
    df = project.get_data()
    if not project.is_regresion():
        df[project.get_variable_a_predecir()] = df[project.get_variable_a_predecir()].astype("object")
    
    #df = df.copy()
    df2 = df.copy()
    for col in np.setdiff1d(df2.columns,project.get_variable_a_predecir()):
        df2[col] = df2[col].astype("object")
        
        if len(df2[col].value_counts().tolist())==0:
            df[col] = df[col].astype("object")
        elif df2[col].value_counts().tolist()[0] <= 3:
            df[col] = df[col].astype("object")

    if request.method == 'POST':
        form = ChangeDataTypeForm(request.POST, columns=list(df.columns),dict_initial=df.dtypes)
        if form.is_valid():
            try:
                df2 = df.copy()
                for col in df.columns:
                    if "datetime" in form.cleaned_data[col]:
                        df2[col] = pd.to_datetime(df2[col]).dt.normalize()
                    else:
                        df2[col] = df2[col].astype(form.cleaned_data[col])
                if (not (df2.dtypes == df.dtypes).all()):  
                    project.tiposDatosProcesados.save('project_{}_dtypes.json'.format(id_project), ContentFile(df2.dtypes.astype(str).to_json()))
                project.project_state = 3
                project.save()
                return redirect('/initProject/analisisDescriptivo/tratarNa/'+str(id_project))
            except:
                context["errors"] = "Los tipos de datos a los que intentas cambiar no son posibles.Por favor cambia si quieres los datos en bruto en un fichero de texto"
    
    context["df"] = shuffle(df).head().to_dict(orient="records")
    context["dataTypes"] = df.dtypes 
    context["form"] = ChangeDataTypeForm(columns=list(df.columns),dict_initial=df.dtypes)
    
    return render(request,"projectChildren/analisisDescriptivo/confirmarDatos.html",context=context)


    ######################################################################################################
    #####################################===TRATAR NAs===#################################################
    ######################################################################################################

def tratarNa(request,id_project,cambiarVar=None):
    project= utils.isUserLoggedIn_or_hisProject(request,id_project)
    if type(project)!=type(Project()):
        return redirect('/projects/')
    
    if not cambiarVar:
        if project.project_state<3:
            return redirect('/initProject/analisisDescriptivo/confirmarDatos/'+str(id_project))
        elif project.project_state>3:
            return redirect('/initProject/analisisDescriptivo/visualizarDatos/'+str(id_project))

    
    
    context = {}
    context['proyecto'] = project
    context["tipo"]=project.get_tipo_prediccion()
    
    df = project.get_data(original=True)
    print(df.isna().sum())
    na_columns = df.columns[df.isna().sum() > 0].tolist()
    
    INPUTS = np.setdiff1d(df.columns,project.get_variable_a_predecir())
    
    INPUTS_CAT = df[INPUTS].select_dtypes(include=['object']).columns.values.tolist()
    
    n_obsevaciones = df.shape[0]
    
    columnas_a_borrar = {}
    
    for col in INPUTS_CAT:
        
        if len(df[col].value_counts().tolist())==0:
            columnas_a_borrar[col] = "Representatividad"
        elif (df[col].value_counts().tolist()[0] <= n_obsevaciones*0.1) :
            columnas_a_borrar[col] = "Representatividad"
        elif (len(df[col].unique()) ==1):
            columnas_a_borrar[col] = "1Val"
    
    if (len(na_columns)==0) & (len(columnas_a_borrar.keys())==0):
        project.project_state=4
        project.version_datos_a_usar= datetime.now()
        project.save()
        return redirect('/initProject/analisisDescriptivo/visualizarDatos/'+str(id_project))
    
    
    dict_for_form_multiple_types = {}
    dict_with_data = {}
    for nacol in na_columns:
        missing_val = np.round((df[nacol].isna().sum()*100)/df.shape[0],2)
        dict_with_data[nacol] = {
            "missingVal": missing_val,
            "type":df[nacol].dtype,
        }
    
        if missing_val>=40:
            dict_for_form_multiple_types[nacol] = {
                "tipo":df[nacol].dtype,
                "inicial":"delCol"
                }
        elif missing_val<20:
            dict_for_form_multiple_types[nacol] = {
            "tipo":df[nacol].dtype,
            "inicial":"del"
            }
        else:
            if str(df[nacol].dtype)=="object":
                dict_for_form_multiple_types[nacol] = {
                "tipo":df[nacol].dtype,
                "inicial":"labelMostUsed"
                }
            else:
                dict_for_form_multiple_types[nacol] = {
                "tipo":df[nacol].dtype,
                "inicial":"median"
                }
    
    for col,value in columnas_a_borrar.items():
        dict_for_form_multiple_types[col] = {
            "tipo":"delete",
            "inicial":"delCol"
        }
    
    

    if request.POST:
        form = TratarNA(request.POST,columns=dict_for_form_multiple_types)
        
        if form.is_valid():
            df2 = df.copy()
            
            diccionario_guardar = {}
            
            for col,_ in form.get_fields():

                if form.cleaned_data[col]=='0':                 # assign a 0
                    df2[col]=df2[col].fillna("0")               # assign a 0
                    diccionario_guardar[col] = {
                        "tipo":form.cleaned_data[col],
                        "valor":0
                    }
                elif form.cleaned_data[col]=='median':          # median value
                    df2[col]=df2[col].fillna(df2[col].median()) # median value
                    diccionario_guardar[col] = {
                        "tipo":form.cleaned_data[col],
                        "valor":df2[col].median()
                    }
                elif form.cleaned_data[col]=='del':             # eliminate rows
                    df2.dropna(subset=[col],inplace=True)       # eliminate rows
                    diccionario_guardar[col] = {
                        "tipo":form.cleaned_data[col],
                        "valor":0
                    }
                elif form.cleaned_data[col]=='labelMostUsed':   # most used label
                    df2[col]=df2[col].fillna(df2[col].mode()[0])# most used label
                    diccionario_guardar[col] = {
                        "tipo":form.cleaned_data[col],
                        "valor":df2[col].mode()[0]
                    }
                elif form.cleaned_data[col]=="delCol":          # eliminate column
                    df2.drop(columns=[col],inplace=True)        # eliminate column
                    diccionario_guardar[col] = {
                        "tipo":form.cleaned_data[col],
                        "valor":0
                    }
                else:                                           # Do nothing
                    pass
            
            if not project.is_regresion():
                df2[project.get_variable_a_predecir()] = df2[project.get_variable_a_predecir()].astype("object")
            diccionario_guardar_bueno = json.dumps(diccionario_guardar)
            project.treat_na_dict.save('project_{}_dtypes.json'.format(id_project), ContentFile(diccionario_guardar_bueno))
            
            project.archivoDatosNA.save('project_{}_data.csv'.format(id_project), ContentFile(df2.to_csv(index=False)))
            project.tiposDatosProcesados.save('project_{}_dtypes.json'.format(id_project), ContentFile(df2.dtypes.astype(str).to_json()))
            project.project_state=4
            project.version_datos_a_usar= datetime.now()
            project.save()
            
            return redirect('/initProject/analisisDescriptivo/visualizarDatos/'+str(id_project))        
    
    dict_for_template = {}
    form = TratarNA(columns=dict_for_form_multiple_types)
    for col,formulario in form.get_fields():
        if col not in columnas_a_borrar.keys():
            dict_for_template[col]={
                "var":col,
                "form":formulario,
                "missingVal":str(dict_with_data[col]["missingVal"])+"%",
                "type":dict_with_data[col]["type"],
            }
        else:
            if columnas_a_borrar[col] == "Representatividad":
                dict_for_template[col]={
                    "var":col,
                    "form":formulario,
                    "missingVal":"Demasiada poca representatividad de cada categoría",
                    "type":"object",
                }
            else:
                dict_for_template[col]={
                    "var":col,
                    "form":formulario,
                    "missingVal":"Solo 1 único valor en la columna",
                    "type":"object",
                }
    context["formWithData"] =dict_for_template
    
    return render(request,"projectChildren/analisisDescriptivo/tratarNAN.html",context=context)


    ######################################################################################################
    ##############################===Visualización de Datos===############################################
    ######################################################################################################
    
    
def dataViz(request,id_project,revisit=None):
    project= utils.isUserLoggedIn_or_hisProject(request,id_project)
    if type(project)!=type(Project()):
        return redirect('/projects/')
    
    if not revisit:
        if project.project_state<4:
            return redirect('/initProject/analisisDescriptivo/tratarNa/'+str(id_project))
        elif project.project_state>4:
            return redirect('/modelsProject/elegirModelos/'+str(id_project))
    
    context = {}
    context["proyecto"] = project
    context["tipo"]=project.get_tipo_prediccion()
    
    df = project.get_data()
    
    variables = df.columns.to_list()
    choices = [(col, col) for col in variables]
    
    if request.POST:
        variable1 = request.POST.getlist("var_1")[0]
        variable2 = request.POST.getlist("var_2")[0]
        
        fig = plotlyDescriptivo.plotMultiple(df,variable1,variable2,project.variable_a_predecir)
        context["figure"]=fig.to_html()
        context["formulario"]=FormularioVariables(choices=choices,var_1_initial=variable1,var_2_initial=variable2)
    else:
        fig = plotlyDescriptivo.plotMultiple(df,variables[0],variables[1],project.variable_a_predecir)
        context["figure"]=fig.to_html()
        context["formulario"]=FormularioVariables(choices=choices,var_1_initial=variables[0],var_2_initial=variables[1])

    return render(request,'projectChildren/analisisDescriptivo/visualizacionDatosPlotly.html',context=context)













