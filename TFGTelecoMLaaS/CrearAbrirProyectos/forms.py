from django import forms
from .models import Project
from Autenticacion.models import Usuario

class ProjectForm(forms.ModelForm):
    
    class Meta:
        model = Project
        fields = ['projectName', 'archivoDatos']
    
    def clean_document(self):
        document = self.cleaned_data['archivoDatos']
        if not document.name.endswith(('.csv', '.xlsx')):
            raise forms.ValidationError('File must be a .csv or .xlsx file')
        return document
            
def handle_uploaded_file(f):
    print(f)
    
    
class DeleteProjectsForm(forms.Form):
    modelos_a_eliminar = forms.ChoiceField(choices=[],widget=forms.CheckboxSelectMultiple)
    
    def __init__(self,*args,**kwargs):
        choices = kwargs.pop('choices',[])
        super().__init__(*args,**kwargs)
        self.fields["modelos_a_eliminar"].choices = choices