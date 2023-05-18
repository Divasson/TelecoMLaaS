# TFGMLaaSTeleco
Proyecto de Machine Learning as a Service de la carrera de Ingeniería de Telecomunicaciones con Business Analytics

## Para instalar
0.-Tener instalado Python en el ordenador. Si Python no está instalado, se puede descargar en la tienda de Microsoft (o escribiendo ```python``` en la línea de comandos)

1.-Descargar carpeta de datos, junto con el requirements.txt y guardar en local

2.-En la carpeta donde se han descargado los datos (y está el archivo requirements.txt), hacer click derecho y pinchar en Abrir en Terminal

3.-En el terminal, escribir lo siguiente y darle a enter: ```pip install -r requirements.txt```

En el caso de que el código de antes da problemas, abra la aplicación WINDOWS POWERSHELL como Administrador y ponga: ```New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
-Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force```.

Posteriormente, vuelva a intentar el código ```pip install -r requirements.txt```

## Para encenderlo
Una vez lo anterior haya terminado, para encenderlo, tienes que escribir lo siguiente: 

1.-Primero escribir: ```cd .\TFGTelecoApp\TFGTelecoMLaaS\```

2.-Después escribir: ```python manage.py runserver```

3.-Finalmente abrir el navegador y poner en la barra de búsqueda: ```localhost:8000```
