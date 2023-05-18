from django.contrib import admin
from django.urls import include, path
from .views import *


urlpatterns = [
    path("",seeProyects,name="menuProyectos"),   
    path("createProject/",create_project,name="crearProyecto"),   
    path("deleteProjects/",deleteProyects,name="borrarProyectos"),   
]
