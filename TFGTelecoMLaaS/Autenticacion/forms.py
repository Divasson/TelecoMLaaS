from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import authenticate
from .models import Usuario

class CustomInput(forms.TextInput):
    attrs = {'class': 'form-control custom-input'}

class FormularioRegistro(UserCreationForm):
    email = forms.EmailField(max_length=100)

    
    class Meta:
        model = Usuario
        fields = ('email','username','password1','password2')
        
    def __init__(self, *args, **kwargs):
        super(FormularioRegistro, self).__init__(*args, **kwargs)
        self.fields['username'].widget = forms.TextInput(attrs={
            'class': 'form-control custom-input',
        })
        self.fields['email'].widget = forms.TextInput(attrs={
            'class': 'form-control custom-input',
        })
        self.fields['password1'].widget = forms.TextInput(attrs={
            'class': 'form-control custom-input',
            "type": "password",
        })
        self.fields['password2'].widget = forms.TextInput(attrs={
            'class': 'form-control custom-input',
            "type": "password",
        })


class FormularioAuthUsuario(forms.ModelForm):
    password = forms.CharField(label="Password", widget=forms.PasswordInput())
    
    class Meta:
        model = Usuario
        fields = ('email', 'password')
        
    def __init__(self, *args, **kwargs):
        super(FormularioAuthUsuario, self).__init__(*args, **kwargs)
        self.fields['email'].widget = forms.TextInput(attrs={
            'class': 'form-control custom-input',
        })
        self.fields['password'].widget = forms.TextInput(attrs={
            'class': 'form-control custom-input',
            "type": "password"
        })

    def clean(self):
        if self.is_valid():
            email = self.cleaned_data['email']
            password = self.cleaned_data['password']
            if not authenticate(email=email, password=password):
                raise forms.ValidationError('Invalid Login')
