from django.contrib import admin
from .models import ModelosMachineLearning
# Register your models here.

class ModeloAdmin(admin.ModelAdmin):
    list_display = ['proyecto','modelo','name']
    


admin.site.register(ModelosMachineLearning,ModeloAdmin)