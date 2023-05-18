from django.shortcuts import redirect
from CrearAbrirProyectos.models import Project


def isUserLoggedIn_or_hisProject(request,id_project):
    user = request.user
    print(user)
    if not user.is_authenticated:
        return redirect('/index.html')
    try:
        project = request.user.projects.get(id=id_project)
        return project
    except Project.DoesNotExist:
        return redirect('/index.html')
    
    return project