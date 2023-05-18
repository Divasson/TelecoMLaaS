from django import forms

        
class FormularioModeloPredecir(forms.Form):
    archivo_a_predecir = forms.FileField()
    modeloPredecir = forms.ChoiceField(choices=[])
    
    def __init__(self, *args, **kwargs):
        choices = kwargs.pop('choices',[])
        super().__init__(*args, **kwargs)
        self.fields['modeloPredecir'].choices = choices
        #self.fields['modeloPredecir'].initial = modelo_base
        
