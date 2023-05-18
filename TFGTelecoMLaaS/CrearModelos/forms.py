from django import forms


class FormularioModelosClasificacion(forms.Form):
    modelos = forms.ChoiceField(choices=[],widget=forms.SelectMultiple(attrs={'required': True}))
    
    def __init__(self, choices,*args, **kwargs):
        super(FormularioModelosClasificacion, self).__init__(*args, **kwargs)
        self.fields['modelos'].choices = choices

    
class FormularioModelosRegresion(forms.Form):
    modelos = forms.ChoiceField(choices=[],widget=forms.SelectMultiple(attrs={'required': True}))
    
    def __init__(self,choices, *args, **kwargs):
        super(FormularioModelosRegresion, self).__init__(*args, **kwargs)
        self.fields['modelos'].choices = list(choices)


class FormularioTrainTestValSplit(forms.Form):
    train_test_split = forms.IntegerField(max_value=100,min_value=0,initial = 30,required=True)
    #test_val_split = forms.IntegerField(max_value=100,min_value=0,initial=40,required=True)
    
class FormularioModoFacilModoExperto(forms.Form):
    modo = forms.ChoiceField(choices=[("Normal","Normal"),
                                        ("Experto","Experto")],
                              required=True,
                              label="Hacer predicciones modo")
    
class FormularioEnsembles(forms.Form):
    modelos = forms.ChoiceField(choices=[],widget=forms.SelectMultiple(attrs={'required': True}))
    
    def __init__(self, choices,*args, **kwargs):
        super(FormularioEnsembles, self).__init__(*args, **kwargs)
        self.fields['modelos'].choices = choices
    

#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
#######################################===CLASIFICACIÓN===############################################
#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------   
    
    
class FormularioKNN(forms.Form):
    n_neighbours = forms.IntegerField(max_value=1000,min_value=1,required=True,initial=5,
                                      label="Numero de vecinos a considerar")
    weights=forms.ChoiceField(choices=[("distance","distance"),
                                        ("uniform","uniform")],
                              required=True,
                              label="Puntuación de la distancia (uniforme = normal, distance = ponderar más fuerte los cercanos)")
    metric = forms.ChoiceField(choices=[("euclidean","euclidean"),
                                               ("manhattan","manhattan"),
                                               ('minkowski','minkowski')],
                                     required=True,
                                     label="Tipo de distancia a utilizar en el modelo KNN")
    
class FormularioSVC(forms.Form):
    C = forms.FloatField(max_value=10000.0,
                         min_value=0.0,
                         initial=1.0,
                         required=True,
                         label="Parámetro de regularización de Ridge (l2)")
    gamma=forms.ChoiceField(choices=[("scale","scale"),
                                        ("auto","auto")],
                              required=True,
                              label="Valor de gamma")
    kernel = forms.ChoiceField(choices=[("poly","poly"),
                                        ("rbf","rbf"),
                                        ("sigmoid","sigmoid")],
                              required=True,
                              label="Tipo de Kernel")
    
    
class FormularioLogisticRegresion(forms.Form):
    C = forms.FloatField(max_value=100.0,
                         min_value=0.0,
                         initial=1.0,
                         required=True,
                         label="Parámetro de regularización Inverso")
    solver=forms.ChoiceField(choices=[("lbfgs","lbfgs"),
                                        ("liblinear","liblinear"),
                                        ("newton-cg","newton-cg"),
                                        ("sag","sag"),
                                        ("saga","saga")],
                              required=True,
                              label="Elegir el solver del modelo")
    penalty = forms.ChoiceField(choices=[("l2","l2"),
                                        ("l1","l1"),
                                        ("elasticnet","elasticnet"),
                                        ("None","None")],
                              required=True,
                              label="Tipo de Kernel")
    
    
    
class FormularioRandomForestClassifier(forms.Form):
    
    n_estimators    = forms.IntegerField(max_value=10000,
                                        min_value=10,
                                        initial=100,
                                        required=True,
                                        label="Número de árboles en el Random Forest")
    
    criterion       = forms.ChoiceField(choices=[
                                                ("gini","gini"),
                                                ("entropy","entropy"),
                                                ],
                                            required=True,
                                            label="Tipo de criterio para dividir las hojas")
    
    max_depth    = forms.IntegerField(max_value=100000,
                                        min_value=15,
                                        initial=500,
                                        required=False,
                                        label="Profundidad en el Random Forest")
    
    min_samples_leaf    = forms.IntegerField(max_value=50,
                                                min_value=1,
                                                initial=1,
                                                required=True,
                                                label="Número mínimo de muestras en cada hoja")

#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
##################################===REGRESIÓN===#####################################################
#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------    
    
class FormularioElasticNetCV(forms.Form):
    
    n_alphas    = forms.IntegerField(max_value=200,
                                        min_value=50,
                                        initial=100,
                                        required=True,
                                        label="Número de alphas en el modelo a la hora de entrenar")
    
    eps       = forms.FloatField(max_value=0.01,
                                    min_value=0.00001,
                                    initial=0.001,
                                    required=True,
                                    label="Tamaño del path")
    
    l1_ratio    = forms.IntegerField(max_value=1,
                                        min_value=0,
                                        initial=0.5,
                                        required=True,
                                        label="Parámetro de regularización")
    

class FormularioRandomForestRegressor(forms.Form):
    
    n_estimators    = forms.IntegerField(max_value=10000,
                                        min_value=10,
                                        initial=100,
                                        required=True,
                                        label="Número de árboles en el Random Forest")
    
    criterion       = forms.ChoiceField(choices=[
                                                ("squared_error","squared_error"),
                                                ("absolute_error","absolute_error"),
                                                ("friedman_mse","friedman_mse"),
                                                ("poisson","poisson")],
                                            required=True,
                                            label="Tipo de criterio para dividir las hojas")
    
    max_depth    = forms.IntegerField(max_value=100000,
                                        min_value=15,
                                        initial=500,
                                        required=False,
                                        label="Profundidad en el Random Forest")
    
    min_samples_leaf    = forms.IntegerField(max_value=50,
                                                min_value=1,
                                                initial=1,
                                                required=True,
                                                label="Número mínimo de muestras en cada hoja")
    
    
class FormularioSVR(forms.Form):
    C = forms.FloatField(max_value=10000.0,
                         min_value=0.0,
                         initial=1.0,
                         required=True,
                         label="Parámetro de regularización de Ridge (l2)")
    gamma=forms.ChoiceField(choices=[("scale","scale"),
                                        ("auto","auto")],
                              required=True,
                              label="Valor de gamma")
    kernel = forms.ChoiceField(choices=[("poly","poly"),
                                        ("rbf","rbf"),
                                        ("sigmoid","sigmoid"),
                                        ("linear","linear")],
                              required=True,
                              label="Tipo de Kernel")
    degree    = forms.IntegerField(max_value=7,
                                        min_value=1,
                                        initial=4,
                                        required=True,
                                        label="Número de grados en el modelo")
    
class Formulario_Red_Neuronal(forms.Form):
    hidden_layer_sizes = forms.IntegerField(max_value=15,
                                        min_value=1,
                                        initial=5,
                                        required=True,
                                        label="Número de capas internas en el modelo")
    neurons_per_layer = forms.IntegerField(max_value=200,
                                        min_value=10,
                                        initial=100,
                                        required=True,
                                        label="Número de neuronas en cada una de las capas en el modelo")
    activation_function = forms.ChoiceField(choices=[("relu","relu"),
                                                    ("selu","selu"),
                                                    ("softmax","softmax")],
                                        required=True,
                                        label="Función de activación en las neuronas intermedias")