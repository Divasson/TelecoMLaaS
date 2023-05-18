import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from sklearn.metrics import classification_report, confusion_matrix

def getListaModelosClasificacion(binaryClass=False):
    lista = [
        ("KNN","KNN"),
        ("SVC","SVC"),
        ("Logistic Regression","Logistic Regression"),
        ("Random Forest","Random Forest"),
        ("Neural Network","Neural Network"),
    ]
    return lista

def getListaModelosRegresion():
    lista = [
        ("ElasticNetCV","ElasticNetCV"),
        ("Random Forest","Random Forest"),
        ("KNN","KNN"),
        ("Neural Network","Neural Network"),
        ("Linear Regression","Linear Regression"),
        ("SVR","SVR"),
    ]
    return lista





def ValidacionModeloClasificacion(modelo,X_train,Y_train,X_test,Y_test,labels=None): 
    """
    Plotea todas las métriscas tanto para el conjunto de train como de test y devuelve la matriz de confusion para el conjunto de test
    
    
    Parameters
    ----------
    modelo : Modelo de clasificacion
    X_train: Datos de entrenamiento
    Y_train: Etiquetas de entrenamiento
    X_test : Datos de test
    Y_test : Etiquetas de test
    
    Returns
    -------
    fig: Grafica de la matriz de confusion para el conjunto de test
    
    Examples
    --------
    >>> ValidacionModeloClasificacion(modelo,X_train,Y_train,X_test,Y_test)
    """
    
    pred_train = modelo.predict(X_train)
    pred_test  = modelo.predict(X_test)
    
    
    class_report_train = classification_report(y_true=Y_train,y_pred=pred_train)
    class_report_test  = classification_report(y_true=Y_test,y_pred=pred_test)
    print("TRAIN:")
    print(class_report_train)
    print("TEST:")
    print(class_report_test)
    
    if labels!=None:
        x = labels
        y = labels
    else:
        x = [str(x) for x in list(Y_train.unique())]
        y = [str(x) for x in list(Y_train.unique())]
    
    z = confusion_matrix(
        Y_test.argmax(axis=1), pred_test.argmax(axis=1)
        )
    

    fig = ff.create_annotated_heatmap(
            x               = x,
            y               = y,
            z               = z,
            annotation_text = z,
            hoverinfo       = 'z',
            colorscale      = px.colors.sequential.Viridis
    )
    fig.update_layout(title="Matriz de confusion para el conjunto de test")
    return fig,class_report_train,class_report_test
