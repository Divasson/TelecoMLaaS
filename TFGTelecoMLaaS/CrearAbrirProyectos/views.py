from django.shortcuts import render, redirect
from .forms import *

# Create your views here.

def create_project(request):
    
    user = request.user
    print(user)
    if not user.is_authenticated:
        return redirect('/index.html')
    
    
    if request.method == 'POST':
        form = ProjectForm(request.POST, request.FILES)
        
        if form.is_valid():
            print("El id de usuario es "+str(request.user.id))
            #
            project = form.save(commit=False)            
            project.usuario = request.user # add the current user as a related user
            project.save()
            project.set_columnas_originales(list(project.get_data().columns))
            project.save()
            return redirect('/projects/')
    else:
        form = ProjectForm()
    return render(request, 'crearProyectos.html', {'form': form})


def seeProyects(request):
    user = request.user
    print(user)
    if not user.is_authenticated:
        return redirect('/index.html')
    
    projects = request.user.projects.all()[::-1]
    print(projects)
    return render(request,'menuProyectos.html',{'projects': projects})
    
def deleteProyects(request):
    user = request.user
    print(user)
    if not user.is_authenticated:
        return redirect('/index.html')
    
    projects = request.user.projects.all()[::-1]
    if len(projects)<1:
        return redirect("/projects/")
    
    choices = [(project.get_nombre_proyecto(),project.get_nombre_proyecto()) for project in list(projects)]
    if request.POST:
        modelos_seleccionados = list(request.POST.getlist('modelos_a_eliminar'))
        if len(modelos_seleccionados)<1:
            return redirect("/projects/")
        else:
            for project in projects:
                if project.get_nombre_proyecto() in modelos_seleccionados:
                    project.borrar_todos_archivos_vinculados()
                    project.delete()
            return redirect("/projects/")

    context = {}
    
    context["formulario"] = DeleteProjectsForm(choices=choices)
    
    
    return render(request,"deleteProyectos.html",context=context)
        
    
    

