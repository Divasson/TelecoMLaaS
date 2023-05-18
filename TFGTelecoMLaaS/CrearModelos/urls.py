from django.contrib import admin
from django.urls import include, path
from .views import *


urlpatterns = [
    path("<int:id_project>",inicio,name="inicio"),   
    path("<int:id_project>/<int:cambiar>",inicio,name="inicio"),   
    path("elegirModelos/<int:id_project>",elegirModelos,name="elegirModelos"),   
    path("elegirModelos/<int:id_project>/<int:cambiar>",elegirModelos,name="elegirModelos"),   
    path("preprocesado/<int:id_project>",preprocesado,name="preprocesado"),   
    path("preprocesado/<int:id_project>/<int:cambiar>",preprocesado,name="preprocesado"),   
    path("entrenarModelos/<int:id_project>",entrenarModelosOptuna,name="entrenarModelos"),
    path("entrenarModelos/<int:id_project>/<int:cambiar>",entrenarModelosOptuna,name="entrenarModelos"),
    path("entrenarModelos/<int:id_project>/<str:nombremodelo>",entrenarModelosOptuna,name="entrenarModelos"),
    path("entrenarModelosConParams/<int:id_project>/<str:nombremodelo>",entrenarModelosParams,name="entrenarModelosOptuna"),
    path("entrenarModelosConParams/<int:id_project>/<str:nombremodelo>/<int:cambiar>",entrenarModelosParams,name="entrenarModelosOptuna"),
    path("crearEnsemble/<int:id_project>",crearEnsemble,name="crearEnsemble"),
]
