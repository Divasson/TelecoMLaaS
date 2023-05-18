from django.contrib import admin
from django.urls import include, path
from .views import *
urlpatterns = [
    path('',landing_page,name='landing_page'),
    path('index.html',landing_page,name='landing_page'),
    
]
