from django.contrib import admin
from django.urls import include, path
from .views import *


urlpatterns = [
    path("<int:id_project>",vizModelo,name="vizModelo"),   
    path("<int:id_project>/<str:nombremodelo>",vizModelo,name="vizModelo"),   
    
]
