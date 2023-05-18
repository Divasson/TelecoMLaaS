from django.contrib import admin
from django.urls import include, path
from .views import *
urlpatterns = [
    path('<int:id_project>',subirPrediccion,name='subirPrediccion'),
    path('makePred/<int:id_project>',hacerPrediccion,name='hacerPrediccion'),
]
