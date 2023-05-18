from django.shortcuts import render,redirect
from django.contrib.auth import login,authenticate,logout
from .forms import FormularioRegistro,FormularioAuthUsuario


def registro(request):
    context = {}
    if request.POST: # POST request
        form = FormularioRegistro(request.POST)

        if form.is_valid():
            form.save()
            email = form.cleaned_data['email']
            rawPassword = form.cleaned_data['password1']
            usuario = authenticate(email=email,password= rawPassword)
            login(request,usuario)
            return redirect('/projects/')
        else:
            context['form']=form
    else: # GET request
        form = FormularioRegistro()
        context['form']=form
    return render(request,"Register.html",context)

def logout_view(request):
    logout(request)
    return redirect('/')

def login_view(request):
    context = {}
    user = request.user
    print(user)
    if user.is_authenticated:
        return redirect('/projects/')
    
    if request.POST:
        form = FormularioAuthUsuario(request.POST)
        if form.is_valid():
            email = request.POST['email']
            password = request.POST['password']
            user = authenticate(email = email, password =password)
            
            if user:
                login(request,user)
                return redirect('/projects/')
    else:
        form = FormularioAuthUsuario()
    
    context['form'] = form
    return render(request,"Login.html",context)


