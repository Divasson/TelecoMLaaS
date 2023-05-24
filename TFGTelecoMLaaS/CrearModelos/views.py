from django.shortcuts import redirect, render

from AnalizarDatos import utils
from AnalizarDatos.utils import utils

from .forms import *
from .models import ModelosMachineLearning, Project
from .utils import utils as utils_list_modelos


# Create your views here.
def inicio(request,id_project,cambiar=None):
    project= utils.isUserLoggedIn_or_hisProject(request,id_project)
    if type(project)!=type(Project()):
        return redirect('/projects/')
    
    
    if not cambiar:
        if project.project_state>4:
            if project.project_state>7:
                return redirect('/vizProject/'+str(id_project))
            else:
                return redirect('/modelsProject/elegirModelos/'+str(id_project))
        elif project.project_state<4:
            return redirect('/initProject/analisisDescriptivo/seleccionarVariable/'+str(id_project))
    
    
    if request.POST:
        modo = request.POST.getlist("modo")[0]
        if modo=="Experto":
            project.project_state = 5
            project.save()
            return redirect('/modelsProject/elegirModelos/'+str(id_project))
        else:
            listaModelos = 0
            if project.is_regresion():
                listaModelos = utils_list_modelos.getListaModelosRegresion()
            else:
                listaModelos = utils_list_modelos.getListaModelosClasificacion(binaryClass=project.is_binary_model())
                
            listaModelosBien = [x[0] for x in listaModelos]
            project.set_modelos_seleccionados(listaModelosBien)
            project.set_train_test_val_splits(30,0)
            project.preprocesar_datos_y_guardarlos()
            project.save()
            listaModelosMalos = []
            for modelo in listaModelosBien:
                try:
                    guardarModelo_Optuna(proyecto=project,nombreModelo=modelo)
                except:
                    listaModelosMalos.append(modelo)
                    
            if len(listaModelosMalos)>0:
                listaModelosSeleccionadosBien = [ele for ele in listaModelosBien if ele not in listaModelosMalos]
                project.set_modelos_seleccionados(listaModelosSeleccionadosBien)
            
            project.project_state = 8
            project.save()
            
            return redirect("/vizProject/"+str(project.id))
    
    
    context = {}
    context["proyecto"]=project
    
    context["formularioModo"]=FormularioModoFacilModoExperto()
    context["tipo"] = project.get_tipo_prediccion()
    
    
    return render(request,"modoFacilModoExperto.html",context=context)

def crearEnsemble(request,id_project):
    project= utils.isUserLoggedIn_or_hisProject(request,id_project)
    if type(project)!=type(Project()):
        return redirect('/projects/')
    
    modelos_entrenados = project.get_list_modelos_entrenados()

    if len(modelos_entrenados)<1:
        return redirect('/modelsProject/'+str(id_project))
        
    context = {}
    context["proyecto"]=project
    context["tipo"] = project.get_tipo_prediccion()
    context["ensembleYaElegido"] = False
    
    dict_modelos_entrenados = {}
    for modelo_entrenado_nombre in modelos_entrenados:
        modelo_clase = ModelosMachineLearning.objects.filter(proyecto=project,name=modelo_entrenado_nombre).first()
        if not str(modelo_clase.name).startswith("Ensemble"):
            dict_modelos_entrenados[modelo_entrenado_nombre] = modelo_clase.get_metrica_modelo()
        else:
            context["ensembleYaElegido"] = True
        
    if project.is_regresion():
        dict_modelos_entrenados = dict(sorted(dict_modelos_entrenados.items(),key=lambda x:x[1],reverse=False))
    else:
        dict_modelos_entrenados = dict(sorted(dict_modelos_entrenados.items(),key=lambda x:x[1],reverse=True))
    
    
    choices = [(col,str(col)+"-"+str(round(dict_modelos_entrenados[col],2))) for col in list(dict_modelos_entrenados.keys())]
    context["formulario"] = FormularioEnsembles(choices=choices)
    
    context["is_ensemble"] = True
    
    if request.POST:        
        modelos_seleccionados = request.POST.getlist("modelos")
        if len(modelos_seleccionados)>1:
            context["Ensemble_1_selected"] = False
            guardarModelo_Ensemble(proyecto=project,listaModelos=modelos_seleccionados)
            return redirect('/vizProject/'+str(id_project))
        else:
            context["Ensemble_1_selected"] = True

    return render(request,"crearModelos/crearEnsemble.html",context=context)



    

def elegirModelos(request,id_project,cambiar=None):
    project= utils.isUserLoggedIn_or_hisProject(request,id_project)
    if type(project)!=type(Project()):
        return redirect('/projects/')
    
    if not cambiar:
        if project.project_state>5:
            if project.project_state>6:
                return redirect('/vizProject/'+str(id_project))
            else:
                return redirect('/modelsProject/preprocesado/'+str(id_project))
        elif project.project_state<5:
            return redirect('/initProject/analisisDescriptivo/seleccionarVariable/'+str(id_project))
    
    if request.POST:
        modelos = request.POST.getlist("modelos")
        project.set_modelos_seleccionados(modelos)
        project.project_state = 6
        project.save()
        
        return redirect('/modelsProject/preprocesado/'+str(id_project))
        
    context = {}
    context["proyecto"]=project
    context["tipo"] = project.get_tipo_prediccion()
    if project.is_regresion():
        context["formulario"]=FormularioModelosRegresion(choices = utils_list_modelos.getListaModelosRegresion())
    else:
        df = project.get_data()
        binary = False
        
        context["formulario"]=FormularioModelosClasificacion(choices = utils_list_modelos.getListaModelosClasificacion(binaryClass=project.is_binary_model()))
        
    listaModelos = project.is_modelos_seleccionados()
    if listaModelos:
        context["modelosYaElegidos"] = 1
    else:
        context["modelosYaElegidos"] = 0
        
    return render(request,"crearModelos/elegirModelos.html",context=context)

def preprocesado(request, id_project,cambiar=None):
    project= utils.isUserLoggedIn_or_hisProject(request,id_project)
    if type(project)!=type(Project()):
        return redirect('/projects/')

    if not cambiar:
        if project.project_state>6:
            if project.project_state>7:
                return redirect('/vizProject/'+str(id_project))
            else:
                return redirect('/modelsProject/entrenarModelos/'+str(id_project))
        elif project.project_state<6:
            return redirect('/modelsProject/elegirModelos/'+str(id_project))    
    
    if request.POST:
        formulario = FormularioTrainTestValSplit(request.POST)
        if formulario.is_valid():
            train_test_split = formulario.cleaned_data["train_test_split"]
            #test_val_split = formulario.cleaned_data["test_val_split"]
            project.set_train_test_val_splits(train_test_split,0)
            project.preprocesar_datos_y_guardarlos()
            project.project_state = 7
            project.save()
            return redirect('/modelsProject/entrenarModelos/'+str(id_project))
    
    context = {}
    context["proyecto"]=project
    context["formulario"]=FormularioTrainTestValSplit()
    context["tipo"] = project.get_tipo_prediccion()
    context["preprocesado_hecho"] = project.is_train_test_split_done()
    
    return render(request,"crearModelos/preprocesarDatosSplit.html",context=context)


def entrenarModelosOptuna(request,id_project,nombremodelo=None,cambiar=False):
    project= utils.isUserLoggedIn_or_hisProject(request,id_project)
    if type(project)!=type(Project()):
        return redirect('/projects/')
    listaModelos = list(project.get_modelos_seleccionados())
    listaModelosEntrenados = project.get_list_modelos_entrenados()
    
    if not cambiar:
        if project.project_state>7:
            return redirect('/vizProject/'+str(id_project))
        else:
            if checkAllModelsTrained(project=project):
                return redirect('/vizProject/'+str(id_project))

    if nombremodelo==None:
        nombremodelo=listaModelos[0]
        
    siguienteModelo = get_siguienteModelo(nombremodelo,listaModelos)
    context=get_context_entrenarModelos(project,nombremodelo,siguienteModelo,listaModelos,listaModelosEntrenados)    
    
    if request.POST:    
        if nombremodelo!=None: 
            guardarModelo_Optuna(proyecto=project,nombreModelo=nombremodelo)
            nombremodelo = actualizarNextModelo(nombreModeloActual=nombremodelo,listaModelos=listaModelos)
            siguienteModelo = get_siguienteModelo(nombremodelo,listaModelos)
            context=get_context_entrenarModelos(project,nombremodelo,siguienteModelo,listaModelos,listaModelosEntrenados,True)
            
            if checkAllModelsTrained(project=project):
                project.project_state = 8
                project.save()
                #return redirect('/vizProject/'+str(id_project))
       
    return render(request,"crearModelos/entrenarModelosParams_o_no/entrenarModelosOptuna.html",context=context)

def entrenarModelosParams(request,id_project,nombremodelo,cambiar=False):
    project= utils.isUserLoggedIn_or_hisProject(request,id_project)
    if type(project)!=type(Project()):
        return redirect('/projects/')
    listaModelos = list(project.get_modelos_seleccionados())
    listaModelosEntrenados = project.get_list_modelos_entrenados()
    siguienteModelo = get_siguienteModelo(nombremodelo,listaModelos)

    if not cambiar:
        if checkAllModelsTrained(project=project):
            return redirect('/vizProject/'+str(id_project))
        
    siguienteModelo = get_siguienteModelo(nombremodelo,listaModelos)
    context=get_context_entrenarModelos(project,nombremodelo,siguienteModelo,listaModelos,listaModelosEntrenados)
        
    if request.POST:
        
        if nombremodelo.lower()=="KNN".lower():
            form = FormularioKNN(request.POST)
            if form.is_valid():
                guardarModelo_ConParams(proyecto=project,
                                        nombreModelo=nombremodelo,
                                        params=form.cleaned_data)
                
        elif nombremodelo.lower()=="SVC".lower():
            form = FormularioSVC(request.POST)
            if form.is_valid():
                guardarModelo_ConParams(proyecto=project,
                                        nombreModelo=nombremodelo,
                                        params=form.cleaned_data)
                                
        elif nombremodelo.lower()=="Logistic Regression".lower():
            form = FormularioLogisticRegresion(request.POST)
            if form.is_valid():
                guardarModelo_ConParams(proyecto=project,
                                        nombreModelo=nombremodelo,
                                        params=form.cleaned_data)
                                
        elif nombremodelo.lower()=="Random Forest".lower():
            if project.is_regresion():
                form = FormularioRandomForestRegressor(request.POST)
            else:
                form = FormularioRandomForestClassifier(request.POST)
            if form.is_valid():
                guardarModelo_ConParams(proyecto=project,
                                        nombreModelo=nombremodelo,
                                        params=form.cleaned_data)
        elif nombremodelo.lower() == "Neural Network".lower():
            form = Formulario_Red_Neuronal(request.POST)
            if form.is_valid():
                guardarModelo_ConParams(proyecto=project,
                                        nombreModelo=nombremodelo,
                                        params=form.cleaned_data)
        elif nombremodelo.lower() == "ElasticNetCV".lower():
            form = FormularioElasticNetCV(request.POST)
            if form.is_valid():
                guardarModelo_ConParams(proyecto=project,
                                        nombreModelo=nombremodelo,
                                        params=form.cleaned_data)
        elif nombremodelo.lower() == "Linear Regression".lower():
            #No hay formulario
            guardarModelo_ConParams(proyecto=project,
                                        nombreModelo=nombremodelo,
                                        params=None)
            
        elif nombremodelo.lower()  == "SVR".lower():   
            form = FormularioSVR(request.POST)
            if form.is_valid():
                guardarModelo_ConParams(proyecto=project,
                                        nombreModelo=nombremodelo,
                                        params=form.cleaned_data)
        ## PARA TODOS LOS MODELOS
        
        (nombremodelo) = actualizarNextModelo(nombreModeloActual=nombremodelo,listaModelos=listaModelos)
        # get context
        siguienteModelo = get_siguienteModelo(nombremodelo,listaModelos)
        context=get_context_entrenarModelos(project,nombremodelo,siguienteModelo,listaModelos,listaModelosEntrenados,True)
        
        if checkAllModelsTrained(project=project):
            project.project_state = 8
            project.save()
            #return redirect('/vizProject/'+str(id_project))
        else:
            return redirect('/modelsProject/entrenarModelos/'+str(id_project)+'/'+nombremodelo)
                
    
    if str(nombremodelo).lower()=="KNN".lower():
        context["formulario"] = FormularioKNN()
    elif str(nombremodelo).lower() == "SVC".lower():
        context["formulario"] = FormularioSVC()
    elif str(nombremodelo).lower()=="Logistic Regression".lower():
        context["formulario"] = FormularioLogisticRegresion()
    elif nombremodelo.lower()=="Random Forest".lower():
        if project.is_regresion():
            context["formulario"] = FormularioRandomForestRegressor()
        else:
            context["formulario"] = FormularioRandomForestClassifier()
    elif nombremodelo.lower() == "Neural Network".lower():
        context["formulario"] = Formulario_Red_Neuronal()
    elif nombremodelo.lower() == "ElasticNetCV".lower():
        context["formulario"] = FormularioElasticNetCV()
    elif nombremodelo.lower()  == "SVR".lower():   
        context["formulario"] = FormularioSVR()
        
    
    
    return render(request,"crearModelos/entrenarModelosParams_o_no/entrenarModelosParams.html",context=context)
    
    
    
    
    
        
    
    
    
    
    
    
    
######################################################################################################
#####################################===FUNCIONES VARIAS===###########################################
######################################################################################################
    

#-----------------------------------------------------------------------------------------------------
#----------------------------------- === GUARDAR MODELOS === -----------------------------------------
#-----------------------------------------------------------------------------------------------------
    
def guardarModelo_ConParams(proyecto,nombreModelo,params):
    modeloObjeto = ModelosMachineLearning()
    modeloObjeto.rellenarModeloConParams(proyecto=proyecto,
                                        nombre_modelo_ml=nombreModelo,
                                        params=params) 
    modeloObjeto.save()
    
def guardarModelo_Optuna(proyecto,nombreModelo):
    modeloObjeto = ModelosMachineLearning()
    modeloObjeto.rellenarModeloOptuna(proyecto=proyecto,
                                    nombre_modelo_ml=nombreModelo) 
    modeloObjeto.save()
    

def guardarModelo_Ensemble(proyecto,listaModelos):
    modeloObjeto = ModelosMachineLearning()
    modeloObjeto.rellenarModeloEnsemble(proyecto=proyecto,
                                        nombres_modelos_ml_entrenados=listaModelos) 
    modeloObjeto.save()
    
#-----------------------------------------------------------------------------------------------------
#-------------------------- === ACTUALIZAR LISTAS Y COMPROBAR === ------------------------------------
#-----------------------------------------------------------------------------------------------------
    
def actualizarNextModelo(nombreModeloActual,listaModelos):
    return listaModelos[(int(listaModelos.index(nombreModeloActual))+1)-len(listaModelos)*int((int(listaModelos.index(nombreModeloActual))+1)/len(listaModelos))]

def checkAllModelsTrained(project):
    if project.is_all_models_trained():
        project.state=8
        project.save()
        return True
    return False

def get_context_entrenarModelos(project,nombremodelo,siguienteModelo,listaModelos,listaModelosEntrenados,modeloRecienEntrenado=False):
    """
    It returns a dictionary with the following keys:
    
    - proyecto
    - modeloSeleccionado
    - siguienteModelo
    - lista_modelos
    - lista_modelosEntrenados
    - tipo
    - modelo_recien_entrenado_confirmacion
    
    The values of the keys are the following:
    
    - proyecto: the project object
    - modeloSeleccionado: the name of the model
    - siguienteModelo: the name of the next model
    - lista_modelos: the list of models
    - lista_modelosEntrenados: the list of trained models
    - tipo: the type of the project (regression or classification)
    - modelo_recien_entrenado_confirmacion:
    
    :param project: the project object
    :param nombremodelo: the name of the model that is being trained
    :param siguienteModelo: The next model to be trained
    :param listaModelos: list of models to be trained
    :param listaModelosEntrenados: list of models that have been trained
    :param modeloRecienEntrenado: Boolean that indicates if the model was just trained, defaults to
    False (optional)
    :return: A dictionary with the following keys:
    """
    context = {}
    context["proyecto"]= project
    context["modeloSeleccionado"]=nombremodelo
    context["siguienteModelo"]=siguienteModelo
    context["lista_modelos"]= listaModelos
    context["lista_modelosEntrenados"] = listaModelosEntrenados
    
    context["tipo"]=project.get_tipo_prediccion()
    
    context["modelo_recien_entrenado_confirmacion"] = modeloRecienEntrenado
    return context
     
def get_siguienteModelo(nombremodelo,listaModelos):
    return listaModelos[(int(listaModelos.index(nombremodelo))+1)-len(listaModelos)*int((int(listaModelos.index(nombremodelo))+1)/len(listaModelos))]
    
    
