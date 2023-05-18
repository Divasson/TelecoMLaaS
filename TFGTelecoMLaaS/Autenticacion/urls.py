from django.urls import path
from .views import *


urlpatterns = [
    path("",login_view,name="login"),   
    path("register/",registro,name="registro"),   
    path('logout/',logout_view,name="Logout"), 
    #path('',VistaLoginRegistro.as_view(),name='autenticacion'),
]
