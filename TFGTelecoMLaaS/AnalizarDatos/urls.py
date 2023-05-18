from django.contrib import admin
from django.urls import include, path
from .views import *
urlpatterns = [
    path('analisisDescriptivo/seleccionarVariable/<int:id_project>',start_project,name='start_project'),
    path('analisisDescriptivo/seleccionarVariable/<int:id_project>/<int:cambiarVar>',start_project,name='start_project_cambiarVar'),
    path('analisisDescriptivo/tipoPrediccion/<int:id_project>',confirmarTipoPrediccion,name='tipoPrediccion'),
    path('analisisDescriptivo/tipoPrediccion/<int:id_project>/<int:cambiarVar>',confirmarTipoPrediccion,name='tipoPrediccion'),
    path('analisisDescriptivo/confirmarDatos/<int:id_project>',confirm_data,name='confirm_data'),
    path('analisisDescriptivo/confirmarDatos/<int:id_project>/<int:cambiarVar>',confirm_data,name='confirm_data'),
    path('analisisDescriptivo/tratarNa/<int:id_project>',tratarNa,name='treat_na'),
    path('analisisDescriptivo/tratarNa/<int:id_project>/<int:cambiarVar>',tratarNa,name='treat_na'),
    path('analisisDescriptivo/visualizarDatos/<int:id_project>',dataViz,name='dataViz'),   
    path('analisisDescriptivo/visualizarDatos/<int:id_project>/<int:revisit>',dataViz,name='dataViz'),   
]
