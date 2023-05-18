import math

from django.shortcuts import redirect, render

from AnalizarDatos.forms import FormularioVariables
from AnalizarDatos.utils import utils
from CrearAbrirProyectos.models import Project
from CrearModelos.models import ModelosMachineLearning

from .utils import fig_creator


# Create your views here.
def vizModelo(request,id_project,nombremodelo=None):
    project= utils.isUserLoggedIn_or_hisProject(request,id_project)
    if type(project)!=type(Project()):
        return redirect('/projects/')
        
    modelos_entrenados = project.get_list_modelos_entrenados()
    
    if len(modelos_entrenados)==0:
        return redirect('/modelsProject/'+str(id_project))
    

    
    dict_modelos_entrenados = {}
    
    for modelo_entrenado_nombre in modelos_entrenados:
        modelo_clase = ModelosMachineLearning.objects.filter(proyecto=project,name=modelo_entrenado_nombre).first()
        dict_modelos_entrenados[modelo_entrenado_nombre] = modelo_clase.get_metrica_modelo()
    
    def custom_sort(obj):
        return (dict_modelos_entrenados[obj[0]], not str(obj[0]).startswith("Ensemble"))
    
    if project.is_regresion():
        dict_modelos_entrenados = dict(sorted(dict_modelos_entrenados.items(), key=custom_sort,reverse=False))
    else:
        dict_modelos_entrenados = dict(sorted(dict_modelos_entrenados.items(), key=custom_sort,reverse=True))

    if nombremodelo==None:
        nombremodelo=list(dict_modelos_entrenados.keys())[0]
    

    modeloML = ModelosMachineLearning.objects.filter(proyecto=project,name=nombremodelo).first()
    
    
    #--------------------------------------------------
    #-------------CONTEXTO GENERAL---------------------
    #--------------------------------------------------
    context = {}
    context["proyecto"]= project
    context["modeloSeleccionado"]=nombremodelo
    context["todos_modelos_entrenados"] = project.is_all_models_trained()
    if project.is_regresion():
        for key in list(dict_modelos_entrenados.keys()):
            number = dict_modelos_entrenados[key]
            if number > 1000:
                #suffixes = ['', 'k', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y']
                suffixes = ['', '1e3', '1e6', '1e9', '1e12', '1e15', '1e18', '1e21', '1e24']
                exp = int(math.log10(number)) // 3 * 3
                formatted_number = "{:.1f} {}".format(number / 10.0**exp, suffixes[int(exp/3)])
                dict_modelos_entrenados[key] = formatted_number
            else:
                dict_modelos_entrenados[key] = number
    else:
        for key in list(dict_modelos_entrenados.keys()):
            dict_modelos_entrenados[key] = round(float(dict_modelos_entrenados[key]),3)
    
    context["dict_modelosEntrenados"] = dict_modelos_entrenados
    
    #--------------------------------------------------
    #--------------CONTEXTO OPTUNA---------------------
    #--------------------------------------------------
    
    context["optuna"] = modeloML.optuna
    if modeloML.optuna:
        study_optuna = modeloML.get_study_optuna()
        context["figura_optuna_entrenamiento"] = fig_creator.study_to_html(study_optuna)[0]
        context["figura_optuna_importanciaParams"] = fig_creator.study_to_html(study_optuna)[1]

    if "Ensemble" not in modeloML.get_nombre_modelo():
        context["model_params"] = modeloML.get_model_params()

    diccionario_plots_normal = {}
    

    columnas_X = project.get_columns_X()
    (X_test_raw,X_test_transformed,y_test_ohe) = project.get_raw_and_transf_test_data_ohe()
        
    context["tipo"]=project.get_tipo_prediccion()
    if project.is_regresion():
        context["nombre_metrica"] = "rmse_test"
        
        ## Parámetros
        context["error_medio"] = modeloML.get_metrica_modelo()
        
        ## Gráficas
        prediccion = modeloML.predict(X_test_transformed)
        diccionario_plots_normal["plot_prediction_reg"] = fig_creator.plot_prediction_reg(y_=y_test_ohe,
                                                                                          y_pred = prediccion)
        diccionario_plots_normal["plot_residuals"] = fig_creator.plot_residuals(y_=y_test_ohe,
                                                                                y_pred=prediccion)
        diccionario_plots_normal["compare_residuals"] = fig_creator.compare_residuals(listaModelos = list(dict_modelos_entrenados.keys()),
                                                                                        X_=X_test_transformed,
                                                                                        y_=y_test_ohe,
                                                                                        project = project,
                                                                                        nombreModeloSeleccionado=nombremodelo,
                                                                                        error_modelo_seleccionado=y_test_ohe-prediccion)

        
        context["normal_plots"] = diccionario_plots_normal
        
        return render(request, 'vizModelos/vizModelosRegresion.html', context)
        
        
    else:
        diccionario_plots_complejo = {}
        context["nombre_metrica"] = "acc_balan_test"
        
        prediction_dict = project.get_prediction_dict()
        
        choices = [(col, col) for col in columnas_X]
        if request.POST:
            variable1 = request.POST.getlist("var_1")[0]
            variable2 = request.POST.getlist("var_2")[0]
        else:
            variable1 = columnas_X[0]
            variable2 = columnas_X[1]
        
        context["formulario_variables"] = FormularioVariables(choices=choices,var_1_initial=variable1,var_2_initial=variable2)
        print("confusion_matrix")
        diccionario_plots_normal["confusion_matrix"] = fig_creator.get_confusion_matrix(model=modeloML,
                                                                                        X_=X_test_transformed,
                                                                                        y_=y_test_ohe,
                                                                                        pred_dict=prediction_dict)
        print("curva roc")
        diccionario_plots_complejo["roc_curve"] = fig_creator.get_roc_curve(model=modeloML,
                                                                          X=X_test_transformed,
                                                                          y=y_test_ohe,    
                                                                          pred_dict=prediction_dict)
        print("plot prediction")
        diccionario_plots_normal["plot_prediction"] = fig_creator.plot_prediction(model=modeloML,
                                                                                  X_raw=X_test_raw,
                                                                                  X_trans=X_test_transformed,
                                                                                  y_=y_test_ohe,
                                                                                  pred_dict=prediction_dict,
                                                                                  X_columns=project.get_columns_X(),
                                                                                  target=project.get_variable_a_predecir(),
                                                                                  variable1=variable1,
                                                                                  variable2=variable2,
                                                                                  proyecto_asociado=project)
        
        
        context["normal_plots"] = diccionario_plots_normal
        context["complex_plots"] = diccionario_plots_complejo
        
        return render(request, 'vizModelos/vizModelosClasificacion.html', context)
    