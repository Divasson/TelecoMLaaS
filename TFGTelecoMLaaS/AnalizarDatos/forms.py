from django import forms

class FormularioVariables(forms.Form):
    var_1 = forms.ChoiceField(choices=[])
    var_2 = forms.ChoiceField(choices=[])

    def __init__(self, choices, *args, **kwargs):
        var_1_initial = kwargs.pop('var_1_initial')
        var_2_initial = kwargs.pop('var_2_initial')
        super(FormularioVariables, self).__init__(*args, **kwargs)
        self.fields['var_1'].choices = choices
        self.fields['var_2'].choices = choices
        self.fields['var_1'].initial = var_1_initial
        self.fields['var_2'].initial = var_2_initial
        
class FormularioVariableaPredecir(forms.Form):
    variablePredecir = forms.ChoiceField(choices=[])
    
    def __init__(self, choices, *args, **kwargs):
        initialVar = kwargs.pop('initialVar')
        super(FormularioVariableaPredecir, self).__init__(*args, **kwargs)
        self.fields['variablePredecir'].choices = choices
        self.fields['variablePredecir'].initial = initialVar
        
class FormularioTipoPrediccion(forms.Form):
    PREDICTION_CHOICES = [
        ('regresion', 'Regresion'),
        ('clasificacion', 'Clasificacion'),
    ]
    prediction_type  = forms.ChoiceField(choices=PREDICTION_CHOICES)
    
    def __init__(self, *args, **kwargs):
        type_of_prediction = kwargs.pop('type_of_prediction')
        super(FormularioTipoPrediccion, self).__init__(*args, **kwargs)
        self.fields['prediction_type'].initial = type_of_prediction
        
class ChangeDataTypeForm(forms.Form):
    DATA_TYPE_CHOICES = [
        ('int64', 'Número Entero'),
        ('float64', 'Número Decimal'),
        ('object', 'Valores categóricos'),
        ('datetime64',"Fecha")
        # ... add other data types as necessary
    ]

    def __init__(self, *args, **kwargs):
        columns = kwargs.pop('columns')
        dict_initial = kwargs.pop('dict_initial')

        super().__init__(*args, **kwargs)
        for column in columns:
            self.fields[column] = forms.ChoiceField(choices=self.DATA_TYPE_CHOICES)
            if "datetime" in str(dict_initial[column]):
                self.fields[column].initial = "datetime64"
            else:
                self.fields[column].initial = dict_initial[column]
            
    def get_fields(self):
        for field_name in self.fields:
            yield (field_name,self[field_name])
            
            
class TratarNA(forms.Form):
    DATA_TYPE_CHOICES = [
        ('0', 'Valor a 0'),
        ('del', 'Eliminar Filas'),
        # ... add other data types as necessary
    ]
    DATA_TYPE_CHOICES_NUM = [
        ('0', 'Valor a 0'),
        ('del', 'Eliminar Filas'),
        ('median','Valor Mediano'),
        ('delCol','Eliminar Columna'),
    ]
    DATA_TYPE_CHOICES_CAT = [
        ('0', 'Valor a 0'),
        ('del', 'Eliminar Filas'),
        ('labelMostUsed','Categoría más presente'),
        ('delCol','Eliminar Columna'),
        # ... add other data types as necessary
    ]
    DATA_TYPE_CHOICES_NON_ESSENTIAL = [
        ('delCol','Eliminar Columna'),
        ('nada','Dejar Columna')
    ]

    
    def __init__(self, *args, **kwargs):
        columns = kwargs.pop('columns')
        #dict_initial = kwargs.pop('dict_initial')

        super().__init__(*args, **kwargs)
        for column,types in columns.items():
            if str(types['tipo'])=="object":
                self.fields[column] = forms.ChoiceField(choices=self.DATA_TYPE_CHOICES_CAT)
                self.fields[column].initial = types["inicial"]
            elif str(types['tipo'])=="delete":
                self.fields[column] = forms.ChoiceField(choices=self.DATA_TYPE_CHOICES_NON_ESSENTIAL)
                self.fields[column].initial = types["inicial"]
            else:
                self.fields[column] = forms.ChoiceField(choices=self.DATA_TYPE_CHOICES_NUM)
                self.fields[column].initial = types["inicial"]
            
    def get_fields(self):
        for field_name in self.fields:
            yield (field_name,self[field_name])