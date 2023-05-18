import numpy as np
import optuna
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import plotly.graph_objects as go
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)

from AnalizarDatos.plotlyDescriptivo import getOrden
from CrearModelos.models import ModelosMachineLearning


listaModelosNotOHE = ["<class 'sklearn.svm._classes.SVC'>"]


def get_confusion_matrix(model,X_,y_,pred_dict):
    
    pred_val = model.predict(X_)
    
       
    x = list(pred_dict.values())
    x = [str(i) for i in x]
    y = x[::-1] # Invertir valores
    
        
    """ if str(type(model)) not in listaModelosNotOHE:
        pred_val = pred_val.argmax(axis=1) """
    
    z = confusion_matrix(y_true=np.vectorize(pred_dict.get)(y_.argmax(axis=1)),
                        y_pred=np.vectorize(pred_dict.get)(pred_val),
                        labels=list(pred_dict.values()))
    z_2 = z[::-1] # Invertir valores
    
    
    print(z)
    print(z_2)
    if not all(value.isdigit() for value in x):
        fig = ff.create_annotated_heatmap(
            x               = x,
            y               = y,
            z               = z_2,
            annotation_text = z_2,
            hoverinfo       = 'z',
            colorscale      = px.colors.diverging.RdYlGn,
            #font_colors     = ["#000000","#ffffff"],
        )
    else:
        fig = ff.create_annotated_heatmap(
            x               = x,
            y               = y,
            z               = z_2,
            annotation_text = z,
            hoverinfo       = 'z',
            colorscale      = px.colors.diverging.RdYlGn,
            #font_colors     = ["#ffffff","#000000"],
        )
    fig.update_layout(title="Matriz de confusión",xaxis_title="Valores predichos",yaxis_title="Valores verdaderos")
    return fig.to_html()



def get_roc_curve(model,X,y,pred_dict):    
    y_scores = np.array(model.predict_proba(X)) 

    # Create an empty figure, and iteratively add new lines
    # every time we compute a new class
    fig = go.Figure()
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    for i in range((y.shape[1])):
        y_true = y[:, i]
        y_score = y_scores[:,i]
               
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_score = roc_auc_score(y_true, y_score)

        name = f"{pred_dict[i]} (AUC={auc_score:.3f})"
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))

    fig.update_layout(
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        #width=700, height=500,
        title="ROC y AUC del modelo"
    )
    return fig.to_html()
    


def plot_prediction(model,X_raw,X_trans,y_,pred_dict,X_columns,target,variable1,variable2,proyecto_asociado):
    
    df = pd.DataFrame(X_raw,columns=X_columns)    
    df[target] = np.vectorize(pred_dict.get)(y_.argmax(axis=1)).astype("object")
    try:
        df = df.astype(dtype=proyecto_asociado.get_raw_dtypes())
    except:
        pass # No hay datos preprocesados

    df["prediction"] = np.vectorize(pred_dict.get)(model.predict(X_trans)).astype("object")
    
    df['correct'] = ['Good prediction' if x else 'Bad prediction' for x in (df[target] == df['prediction'])]
    
    j           = df.copy()
    j["num1"]   = 1
    fig         = 0
    
    colorATener = 2
    
    colores_dict = {
        "Good prediction": px.colors.qualitative.Set1[2],
        "Bad prediction": px.colors.qualitative.Set1[0],
    }
    
    if variable1==variable2:
        if np.issubdtype(df[variable1].dtype, np.number):
            fig = px.histogram(data_frame=df, x=variable1,color="correct",facet_col="correct",color_discrete_map=colores_dict)
            fig.update_layout(title="Histograma de la variable: "+variable1 +" en función de la predicción del modelo",yaxis_title="Numero de muestras observadas")
            return fig.to_html()
        
        else: 
            fig = make_subplots(1, 2, specs=[[{'type':'domain'}, {'type':'domain'}]],
                                subplot_titles=['Good prediction', 'Bad prediction'])
            fig.add_trace(go.Pie(labels=j[j["correct"]=="Good prediction"].groupby(variable1,as_index=False).count()[variable1], 
                                values=j[j["correct"]=="Good prediction"].groupby(variable1,as_index=False).count()["num1"], 
                                textinfo= 'label+percent',
                                textposition='inside',
                                #marker_colors=j[j["correct"]=="Good prediction"].groupby(variable1,as_index=False).count()[variable1].map(getColorsforVariable(df=j,columnaCategorica=variable1,i=colorATener)),
                                marker_colors = px.colors.sequential.Greens,
                                sort=False,
                                name="Good prediction"), 1, 1)
            fig.add_trace(go.Pie(labels=j[j["correct"]=="Bad prediction"].groupby(variable1,as_index=False).count()[variable1], 
                                values=j[j["correct"]=="Bad prediction"].groupby(variable1,as_index=False).count()["num1"], 
                                textinfo= 'label+percent',
                                textposition='inside',
                                #marker_colors=j[j["correct"]=="Bad prediction"].groupby(variable1,as_index=False).count()[variable1].map(getColorsforVariable(df=j,columnaCategorica=variable1,i=colorATener+1)),
                                marker_colors= px.colors.sequential.Reds,
                                sort=False,
                                name="Bad prediction"), 1, 2)
            fig.update_layout(title="Distribución de clases de "+str(variable1)+" según predicción")
            return fig.to_html()        
        
    else:
        if np.issubdtype(df[variable1].dtype, np.number) & np.issubdtype(df[variable2].dtype, np.number):     
            fig = px.scatter(data_frame=j, x=variable1, y=variable2, color="correct",hover_name=target,color_discrete_map=colores_dict)
            fig.update_layout(title="Scatter de "+variable1+" y "+variable2+ " en función de la predicción") 
            return fig.to_html()
        
        elif np.issubdtype(df[variable1].dtype, np.number): 
            fig = px.box(data_frame=j, x=variable1, y=variable2,orientation='h',
                        color="correct",color_discrete_map=colores_dict,category_orders=getOrden(df,"correct"))
            fig.update_layout(title="Box Plot de "+variable1+" por cada valor de "+variable2+ " en función de la predicción del modelo")
            return fig.to_html()
        
        elif np.issubdtype(df[variable2].dtype, np.number): 
            fig = px.violin(data_frame=j, y=variable2, x=variable1,orientation='v',
                        color="correct",color_discrete_map=colores_dict,category_orders=getOrden(df,"correct"))
            fig.update_layout(title="Violin Plot de "+variable2+" por cada valor de "+variable1 + " en función de la predicción del modelo")
            return fig.to_html()
        
        else:   
            fig = px.bar(j, 
                         x=variable1, 
                         y="num1", 
                         color="correct", 
                         barmode="group",
                         color_discrete_map=colores_dict,
                         facet_col=variable2,
                         #category_orders={"day": ["Thur", "Fri", "Sat", "Sun"],"time": ["Lunch", "Dinner"]}
                         )
            fig.update_layout(title="Numero de muestras observadas de "+variable1+" por cada valor de "+variable2,yaxis_title="Numero de muestras observadas")
            return fig.to_html()
    return fig.to_html()


def prediction_boundaries(model,X,y,var1,var2,target,pred_dict,proyectoAsociado):   
    df = pd.DataFrame(X,columns=proyectoAsociado.get_columns_X_transformed())
    
    print(proyectoAsociado.get_columns_X_transformed())
    df[target] = y.argmax(axis=1)
    
    mesh_size = .02
    margin = 0.10
    
    variable1 = "num__"+str(var1)
    variable2 = "num__"+str(var2)
    
    #print(variable1)
    #print(variable2)
    
    x_min, x_max = df[variable1].min() - margin, df[variable1].max() + margin
    y_min, y_max = df[variable2].min() - margin, df[variable2].max() + margin

    xrange = np.arange(x_min, x_max, mesh_size)
    yrange = np.arange(y_min, y_max, mesh_size)
    xx, yy = np.meshgrid(xrange, yrange)
    xx_ravel = xx.ravel()
    yy_ravel = yy.ravel()

    map_pred = 0
    for col in np.setdiff1d(df.columns,[target]):
        if type(map_pred) == type(0):
            if col == variable1:
                map_pred = xx_ravel
            elif col == variable2:
                map_pred = yy_ravel
            else:
                map_pred = np.full(len(xx_ravel),df[col].median())
        else:
            if col == variable1:
                map_pred = np.c_[map_pred,xx_ravel]
            elif col == variable2:
                map_pred = np.c_[map_pred,yy_ravel]
            else:
                map_pred = np.c_[map_pred,np.full(len(xx_ravel),df[col].median())]

    Z = model.predict(map_pred)
    Z = Z.reshape(xx.shape)


    fig = go.Figure(data=[
        go.Contour(
            x=xrange,
            y=yrange,
            z=Z,
            colorscale='RdBu',
            #colorscale=px.colors.qualitative.Plotly,
            line_smoothing=0.8,
            showlegend=False,
            #hovertext=np.vectorize(pred_dict.get)(Z),
            colorbar = dict(
                dtick=len(list(pred_dict.values())),
                tickvals=list(pred_dict.keys()),
                ticktext=list(pred_dict.values()),
                title="Areas de clasificacion",
                
            ),
            contours=dict(
                start=list(pred_dict.keys())[0],
                end =list(pred_dict.keys())[-1],
            )
        )
    ])

    fig.add_trace(
        go.Scatter(
            x=df[variable1], 
            y=df[variable2],
            mode='markers', 
            text=np.vectorize(pred_dict.get)(df[target]),
            marker=dict(
                color = df[target],
                colorscale=px.colors.qualitative.Plotly,
                showscale=False
            )
        )
    )
    fig.update_layout(xaxis_title=var1,
                        yaxis_title=var2,
                        title="Zonas de decisión con las otras variables a la mediana")
    return fig.to_html()

def plot_prediction_reg(model,X_,y_):
    fig = px.scatter(x=y_, y=model.predict(X_), labels={'x': 'Valores reales', 'y': 'Valores predichos'})
    fig.add_shape(
        type="line", line=dict(dash='dash'),
        x0=y_.min(), y0=y_.min(),
        x1=y_.max(), y1=y_.max()
    )
    fig.update_layout(title="Correlación valores reales y predichos en TEST")
    return fig.to_html()

def plot_prediction_reg(y_,y_pred):
    fig = px.scatter(x=y_, y=y_pred, labels={'x': 'Valores reales', 'y': 'Valores predichos'})
    fig.add_shape(
        type="line", line=dict(dash='dash'),
        x0=y_.min(), y0=y_.min(),
        x1=y_.max(), y1=y_.max()
    )
    fig.update_layout(title="Correlación valores reales y predichos en TEST")
    return fig.to_html()

def plot_residuals(model,X_,y_):
    print("Entrando en dibujar residuals")
    error = y_ - model.predict(X_)
    fig = ff.create_distplot([error], ["error"], curve_type='kde',bin_size=40,show_hist=False)
    fig2 = ff.create_distplot([error], ["error"], curve_type = 'normal',bin_size=40)
    normal_x = fig2.data[1]['x']
    normal_y = fig2.data[1]['y']
    fig.add_traces(go.Scatter(x=normal_x, y=normal_y, mode = 'lines',
                            line = dict(color='rgba(0,255,0, 0.6)',
                                        #dash = 'dash'
                                        width = 1),
                            name = 'normal'
                            ))
    fig.update_layout(title="Distribución de los residuos")
    print("Saliendo de dibujar residuals")
    return fig.to_html(include_plotlyjs=False)

def plot_residuals(y_,y_pred):
    print("Entrando en dibujar residuals")
    error = y_ - y_pred
    fig = ff.create_distplot([error], ["error"], curve_type='kde',bin_size=50,show_hist=False)
    fig2 = ff.create_distplot([error], ["error"], curve_type = 'normal',bin_size=40)
    normal_x = fig2.data[1]['x']
    normal_y = fig2.data[1]['y']
    fig.add_traces(go.Scatter(x=normal_x, y=normal_y, mode = 'lines',
                            line = dict(color='rgba(0,255,0, 0.6)',
                                        #dash = 'dash'
                                        width = 1),
                            name = 'normal'
                            ))
    fig.update_layout(title="Distribución de los errores")
    print("Saliendo de dibujar residuals")
    return fig.to_html(include_plotlyjs=False)

def compare_residuals(listaModelos,X_,y_,project,nombreModeloSeleccionado,error_modelo_seleccionado):
    print("Entrando en dibujar comparacion")
    
    fig = go.Figure()
    
    for nombremodelo in listaModelos[::-1]:
        print(nombremodelo)
        if nombremodelo==nombreModeloSeleccionado:
            error = error_modelo_seleccionado
            fig.add_trace(ff.create_distplot([error], [nombremodelo], 
                                                curve_type='kde',
                                                bin_size=40,
                                                show_hist=False).data[0])
            
        else:
            modeloML = ModelosMachineLearning.objects.filter(proyecto=project,name=nombremodelo).first()
            error = y_ - modeloML.predict(X_)
            fig.add_trace(ff.create_distplot([error], [nombremodelo], 
                                                curve_type='kde',
                                                bin_size=40,
                                                show_hist=False,
                                                colors=['silver']).data[0])
        
    fig2 = ff.create_distplot([error], ["error"], curve_type = 'normal',bin_size=40)
    normal_x = fig2.data[1]['x']
    normal_y = fig2.data[1]['y']
    fig.add_traces(go.Scatter(x=normal_x, y=normal_y, mode = 'lines',
                            line = dict(color='rgba(0,255,0, 0.6)',
                                        #dash = 'dash'
                                        width = 1),
                            name = 'normal'
                            ))
    fig.update_xaxes(range=[np.min(error_modelo_seleccionado), np.max(error_modelo_seleccionado)])
    fig.update_layout(title="Comparación de los errores")
    print("Saliendo de dibujar comparacion")
    return fig.to_html(include_plotlyjs=False)
    



def study_to_html(study):
    fig1 = optuna.visualization.plot_optimization_history(study)
    fig1.update_layout(yaxis_title="Metric",xaxis_title="Intento")
    try:
        fig2 = optuna.visualization.plot_param_importances(study)
        return (fig1.to_html(),fig2.to_html())
    except:
        return (fig1.to_html(),None)

