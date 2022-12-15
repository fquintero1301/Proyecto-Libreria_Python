import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
%matplotlib inline

from imblearn.over_sampling import SMOTE
import seaborn as sns
import folium

import plotly as py
import plotly.graph_objects as go
import plotly.express as px


from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree



def visualize_data(x, y):
   '''
    SE NECESITA:
        1. Importar el dataset
        2. Tener claras las siguientes variables:
            x = lista o indice X
            y = lista o indice Y
    '''
   plt.plot(x, y)
   plt.xlabel('X values')
   plt.ylabel('Y values')
   plt.show()

def s_temporal(df, a, y):
    '''
    SE NECESITA:
        1. Importar el dataset
        2. Tener claras las siguientes variables:
            df = El dataframe a utilizar
            a = la medida que se quiere utilizar, ya sea M, W o A
            y = Nombre del eje Y
    '''

    plot = df.resample(a).sum()
    if a == 'M':
        a = 'Meses'
    elif a == 'A':
        a = 'Años'
    elif a == 'W':
        a = 'Semanal'
    plot.plot(xlabel=a, ylabel=y)

def comparacion_stemporal(train, test, prediction, lower_series, upper_series):
    '''
            SE NECESITA:
            1. Importar el dataset
            2. Tener claras las siguientes variables:
            train = datos de train que se utilizan para el modelo
            test = datos de test que se utilizan para poner en practica el modelo
            prediction = la prediccion que se hace del resultado
            lower_series = serie de panda que se basa en la predicción y toma en cuenta pd.Series([:, 0] , test.index)
            upper_series = serie de panda que se basa en la predicción y toma en cuenta pd.Series([:, 1] , test.index)
    '''
    plt.figure(figsize=(15,15))
    plt.plot(train, label='train', lw= 2)
    plt.plot(test, label='actual', lw= 2)
    plt.plot(prediction, label='prediction', lw= 2)
    plt.fill_between(lower_series.index, lower_series, upper_series, color= 'k', alpha=.15)
    plt.title("Prediction vs Actual Numbers")
    plt.legend(loc = 'upper left', fontsize=10)
    plt.show()

def candle_plot(df):
    '''
     Grafica de velas chinas para ver evolución de precios del mercado bursatil, se requieren los parametros del
     precio de entrada "Open", pico más alto del dia "High", el pico de bajada del dia "Low", y el precio del cierre
     "Close", necesitas tener el DataFrame con esas columnas definidas.
    '''
    fig = py.Figure(data=[py.Candlestick(x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])])

    fig.show()

def grafica_creator(df):
    # Se inicializa la figura de plotly
    fig = py.Figure()
    name = str(df['Codigo empresa'].unique())
    # Aquí se agregan los diferentes Scatter, o graficas lineales, que se hace un deploy de 4 graficas manejadas por Botones

    fig.add_trace(
        py.Scatter(x=list(df.index),
                   y=list(df.High),
                   name="High",
                   line=dict(color="#33CFA5")))

    fig.add_trace(
        py.Scatter(x=list(df.index),
                   y=[df.High.mean()] * len(df.index),
                   name="High Average",
                   visible=False,
                   line=dict(color="#33CFA5", dash="dash")))

    fig.add_trace(
        py.Scatter(x=list(df.index),
                   y=list(df.Low),
                   name="Low",
                   line=dict(color="#F06A6A")))

    fig.add_trace(
        py.Scatter(x=list(df.index),
                   y=[df.Low.mean()] * len(df.index),
                   name="Low Average",
                   visible=False,
                   line=dict(color="#F06A6A", dash="dash")))

    # Se agregan las anotaciones, y se crean los botones para cada una de las tablas
    # Las Anotaciones son basicamente el valor promedio, y los umbrales
    high_annotations = [dict(x="2019-01-01",
                             y=df.High.mean(),
                             xref="x", yref="y",
                             text="High Average:f" % df.High.mean(),
                            ax = 0, ay = -40),
                        dict(x=df.High.idxmax(),
                             y=df.High.max(),
                             xref="x", yref="y",
                             text="High Max:f" % df.High.max(),
                            ax = 0, ay = -40)]
    low_annotations = [dict(x="2019-01-01",
                            y=df.Low.mean(),
                            xref="x", yref="y",
                            text="Low Average:f" % df.Low.mean(),
                            ax = 0, ay = 40),
                        dict(x=df.High.idxmin(),
                             y=df.Low.min(),
                             xref="x", yref="y",
                             text="Low Min:f" % df.Low.min(),
                            ax = 0, ay = 40)]
    # Aquí están los botones que permiten manipular si solo quieres ver los valores maximos, minimos, ambos.
    fig.update_layout(
        updatemenus=[
            dict(
                active=0,
                buttons=list([
                    dict(label="None",
                         method="update",
                         args=[{"visible": [True, False, True, False]},
                               {"title": name,
                                "annotations": []}]),
                    dict(label="High",
                         method="update",
                         args=[{"visible": [True, True, False, False]},
                               {"title": name,
                                "annotations": high_annotations}]),
                    dict(label="Low",
                         method="update",
                         args=[{"visible": [False, False, True, True]},
                               {"title": name,
                                "annotations": low_annotations}]),
                    dict(label="Both",
                         method="update",
                         args=[{"visible": [True, True, True, True]},
                               {"title": name,
                                "annotations": high_annotations + low_annotations}]),
                ]),
            )
        ])

    # Set title
    fig.update_layout(title_text=name)

    fig.show()

def grid_creator(data, x, y, hue):
    '''
    Esta funcion permite crear diferentes graficos dependiendo de los parametros que se van pidiend,
    cada una de las graficas sigue un parametro sencillo, primero el data frame, luego el eje x, luego el y,
    por ultimo el hue, te permite ingresar los valores mediante un input para que puedas elegir las columnas
    de tu preferencia

    '''
    fig = plt.figure(constrained_layout=True, figsize=(20,10))
    grid = gridspec.GridSpec(ncols=6, nrows=2, figure=fig)

    #bar plot Horizontal
    ax1 = fig.add_subplot(grid[0, :2])
    ax1.set_title(input(str))
    sns.countplot(data, x= data[input(str)],y=data[input],hue =data[input(str)] , ax=ax1,) #Paid no paid

    #bar plot Vertical
    ax2 = fig.add_subplot(grid[1, :2])
    ax2.set_title('Purpose segmented by Fully Paid/Charged Off')
    bar = sns.barplot(data, x= data[input(str)],y=data[input],hue =data[input(str)] , ax=ax2)
    bar.set_xticklabels(bar.get_xticklabels(),  rotation=90, horizontalalignment='right') #fixing the Names

    #box plot Credit Score
    ax3 = fig.add_subplot(grid[:, 2])
    ax3.set_title('Credit Score')
    sns.boxplot(data.loc[:,input(str)], orient='v', ax = ax3)


    #box plot Monthly payment
    ax4 = fig.add_subplot(grid[:,3])
    ax4.set_title(input('Introduce el titulo de tu Boxplot'))
    sns.boxplot(data[input(str)], orient='v' ,ax=ax4)

    return plt.show()

def Line_Line_bar_party(x, y, y1, label_x="x",
                        label_y="y",
                        label_y1="y1",
                        plotsize=(20, 12),
                        barcolor="grey",
                        linecolor_y="green",
                        linecolor_y1="b"):
    """

    Esta función muestra 3 gráficas, 2 lineales y
    1 lineal combinada con barras, agrupando las vistas comparativas

    Para esta función debes tener instalado las librerias
    de MATPLOTLIB/NUMPY/PANDAS


    Parameters
    ----------
        x : np.array
            eje x
        y : np.array
            eje y
        y1 : np.array
            eje y1
        label_x : str
            Etiqueta del eje x
        label_y : str
            Etiqueta del eje y
        label_y1 : str
            Etiqueta del eje y1
        plotsize : Tuple()
            Tamaño del primer plano
        barcolor : str
            Color de las barras
        linecolor_y : str
            Color de las lineas
        linecolor_y1 : str
            Color de las lineas y1

    Returns
    -------
        Vista comparativa de 2 productos
        según sus ventas, cantidades, compras, o algo asi.

    Notes
    ------

        Función para visualizar la comparación de 2
        columnas, con comportamientos
        similares.

        Ejemplo:
                Tengo 2 productos, y quiero visualizar
                cuanto se ha vendido
                de cada uno en 1 año.

        Francisco Quintero :)


        """


    # Se establece el tamaño de la gráfica y el estilo
    plt.style.use("seaborn-white")
    plt.figure(figsize=plotsize)

    # 1er Linechart con dots
    plt.subplot(2, 3, 1)
    plt.plot(x, y, '-ok');
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.plot(x, y, linestyle="--");

    # 2do Linechart con dots
    plt.subplot(2, 3, 2)
    plt.plot(x, y1, '--');
    plt.xlabel(label_x)
    plt.ylabel(label_y1)
    plt.plot(x, y1, linestyle="--");

    # Linechart+barchart
    plt.subplot(233)
    plt.bar(x, y, color=barcolor)
    plt.plot(x, y, linestyle="--")
    plt.plot(x, y, linestyle="--", color=linecolor_y, label=label_y)
    plt.plot(x, y1, linestyle="--", color=linecolor_y1, label=label_y1)
    plt.xlabel(label_x)
    plt.ylabel(label_y + "_" + label_y1)
    plt.legend(loc="upper right")
    plt.locator_params(axis="x", nbins=len(x))

    plt.show();
    return


def balanced_target(X_train,
                    y_train):
    """
    Esta función realiza un balance del target TRAIN con el
    método SMOTE() y muestra 2 gráficas con el antes y después.

    Para esta función debes tener instalado las librerias
    de MATPLOTLIB/NUMPY/PANDAS/SKLEARN/IMBLEARN


    Recibe los siguientes parámetros:

    Parameters
    ----------
        x_train : np.array
            datos para train
        y_train : np.array
            target para train

    Returns
    -------
        Target balanceado y gráficas

    Notes
    -------

        Para esta función, X_train, y_train deben estar codificados.
        La función fue creada con la finalidad de ahorrar pasos y tener mejor visibilidad
        de el preprocesado en Machine Learning

        Francisco Quintero :)
        """

    smote = SMOTE()

    conteo_balance_target = y_train.value_counts()

    if conteo_balance_target[0] != conteo_balance_target[1]:
        X_train1, y_train = smote.fit_resample(X_train, y_train)
        y_train_balanced = y_train.value_counts()

        plt.figure(figsize=(5, 5))
        plt.subplot(2, 2, 1)
        plt.pie(conteo_balance_target.values,
                labels=conteo_balance_target.index,
                autopct='%1.2f%%')
        p = plt.gcf()
        plt.show()

        plt.figure(figsize=(5, 5))
        plt.subplot(2, 2, 2)
        plt.pie(y_train_balanced.values,
                labels=y_train_balanced.index,
                autopct='%1.2f%%')
        p = plt.gcf()
        plt.show()

        return y_train


def feature_importances_visualization(best_estimator,
                                      X_train,
                                      plotsize=(20, 10)):
    """
    Esta función sirve para visualizar
    el feature importances de un modelo previamente
    entrenado y seleccionado de un GridSearchCV

    Para esta función debes tener instalado las librerias
    de MATPLOTLIB/PANDAS/SKLEARN

    Recibe los siguientes parámetros:

    Parameters
    ----------
        best_estimator : sklearn.model_selection._search.GridSearchCV
            modelo entrenado y seleccionado de un GridSearchCV
        X_train : np.array
            datos para train
        plotsize : Tuple()
            Tamaño del primer plano

    Returns
    -------
        Visualización de los Features importances del modelo entrenado y seleccionado
        de un GridSearchCV mediante un Barchart

    Notes
    -------
        La función fue creada para visualizar cuales son las variables mas usadas
        por un modelo entrenado y seleccionado de un GridSearchCV.

    Francisco Quintero :)
        """


    mejor_estimador = best_estimator.best_estimator_
    fe_i = mejor_estimador.feature_importances_
    df = pd.DataFrame(fe_i, index=X_train.columns).sort_values(0, ascending=False) * 100
    df = df.rename(columns={0: 'Feature_importances'})

    return plt.figure(figsize=plotsize), plt.bar(df.index, df.Feature_importances);

def matrices_comparadas(y, x_test_scaled, y_test, nombre_modelo, y_2, x_test_scaled_2, y_test_2, nombre_modelo_2, size):
    """" Esta función crea dos matrices de confusión y las representa en la misma imagen, para poder compararlas.
    Parámetros:
    y= Target completo para poder sacar las etiquetas del primer modelo
    x_test_scaled= Los parámetros de x_test del primer modelo. NOTA: pueden ser sin escalar, eso no afecta a la función.
    y_test= Target una vez hecho el train_test_split del primer modelo
    nombre_modelo= Primer modelo del que se van a sacar los parámetros
    y_2= Target completo para poder sacar las etiquetas del segundo modelo
    x_test_scaled= Los parámetros de x_test del segundo modelo. NOTA: pueden ser sin escalar, eso no afecta a la función.
    y_test= Target una vez hecho el train_test_split del segundo modelo
    nombre_modelo= Segundo modelo del que se van a sacar los parámetros
    size= tamaño en el cual queremos sacar la imagen. Debe introducirse como tupla, como por ejemplo (7,7)
    """
    fig, (ax1, ax2) = plt.subplots(1,2)
    #primera matriz
    plt.figure(figsize=size)
    cm_labels = np.unique(y)
    predictions = nombre_modelo.predict(x_test_scaled)
    cm_array = confusion_matrix(y_test, predictions)
    cm_array_df = pd.DataFrame(cm_array, index=cm_labels, columns=cm_labels)
    sns.heatmap(cm_array_df, annot=True, annot_kws={"size": 12}, cmap='rocket_r', cbar=False, ax=ax1)
    #segunda matriz
    cm_labels_2 = np.unique(y_2)
    predictions_2 = nombre_modelo_2.predict(x_test_scaled_2)
    cm_array_2 = confusion_matrix(y_test_2, predictions_2)
    cm_array_df_2 = pd.DataFrame(cm_array_2, index=cm_labels_2, columns=cm_labels_2)
    sns.heatmap(cm_array_df, annot=True, annot_kws={"size": 12}, cmap='rocket_r', cbar=False, ax=ax2)
    return plt.show()


def plot_matriz_confusion(y, x_test_scaled, y_test, nombre_modelo, size):
    """" Esta función crea una matriz de confusión y la representa visualmente de una forma elegante y sobria.

    y= Target completo para poder sacar las etiquetas de nuestro modelo
    x_test_scaled= Los parámetros de x_test de nuestro modelo. NOTA: pueden ser sin escalar, eso no afecta a la función.
    y_test= Target una vez hecho el train_test_split del modelo
    nombre_modelo= Modelo del que se van a sacar los parámetros
    size= tamaño en el cual queremos sacar la imagen. Debe introducirse como tupla, como por ejemplo (7,7)
    """
    plt.figure(figsize=size)
    cm_labels = np.unique(y)
    predictions = nombre_modelo.predict(x_test_scaled)
    cm_array = confusion_matrix(y_test, predictions)
    cm_array_df = pd.DataFrame(cm_array, index=cm_labels, columns=cm_labels)
    return sns.heatmap(cm_array_df, annot=True, annot_kws={"size": 12}, cmap='rocket_r', cbar=False)

def piechart_etiquetado(data, size):
    """Esta función nos crea un pie chart con sus correspondientes etiquetas porcentuales, para
    poder saber el % de cada categoría de una forma más sencilla.

    Parámetros:
    data: el dataframe del que vamos a sacar los valores, debe tener las etiquetas como índices y una única columna
    para que la función opere correctamente.
    size: Tamaño que deseamos para la figura. Sus valores deben introducirse como una tupla, como por ejemplo (7,7).
    """
    my_circle = plt.Circle((0, 0), 0.7, color='white')
    plt.figure(figsize=size)
    plt.pie(data.values,
            labels=data.index,
            autopct='%1.2f%%')
    p = plt.gcf()
    p.gca().add_artist(my_circle)
    return plt.show()

def test_transformers(df, cols):  # pasamos como argumentos el dataframe y la lista de columnas
    '''Para esta función necesitamos un dataframe con columnas numéricas
    a las que queramos transformar su distribución hacia una más normal tipo Gauss.
    Esta transformación es necesaria en modelos no supervisados basados en distancias como K-Means,
    previa al estandarizado de los datos.

    Lo ideal antes de pasar los argumentos a la función es tener declaradas dos variables: una que recoja el df
    y otra que nombre las columnas con los datos a transformar en forma de lista.

    Ejemplo:
    # dataframe con los datos sin transformar
    df = pd.read_csv('dataframe_ejemplo')

    # recogemos los nombres de las columnas del dataframe que transformar
    columnas = ["Col1", "Col2", "Col3"]


    PowerTransformer() # instancia del transformador PowerTransformers con los parámetros por defecto (toma valores positivos y negativos)

    QuantileTransformer(output_distribution='normal') # en QT cambiamos el output a normal para ese tipo de distribución


    La función definida sirve para visualizar las posibles distribuciones de los datos, NO guarda los datos transformados.
    Su puesta en marcha responde más a una necesidad de visualizar tal distribución y elegir el mejor preprocesado.

    Para más información sobre estas transformaciones, visita la web
    https://machinelearningmastery.com/quantile-transforms-for-machine-learning/

    Cita necesaria a Yashowardhan Shinde.
    Su perfil en medium: https://yashowardhanshinde.medium.com/
    '''
    # importamos las librerias necesarias para la ejecución de la función

    pt = PowerTransformer()  # instancia del transformador PowerTransformers con los parámetros por defecto (toma valores positivos y negativos)
    qt = QuantileTransformer(
        output_distribution='normal')  # en QT cambiamos el output a normal para ese tipo de distribución

    # definimos el tamaño del plot, pero es recomendable modificarlo para más de 3 columnas

    fig = plt.figure(figsize=(30, 15))

    # definimos j para que la función vaya graficando los plots en cada distribución
    j = 1

    for i in cols:
        # definimos n para que se creen 3 subplots (3 graficos) por cada columna
        n = len(cols)

        # convertimos cada columna a un array de una dimensión
        array = np.array(df[i]).reshape(-1, 1)

        # aplicamos las transformaciones
        y = pt.fit_transform(array)
        x = qt.fit_transform(array)

        # Graficamos la distribución original y cada transformación por cada columna:
        plt.subplot(n, 3, j)
        sns.histplot(array, bins=50, kde=True)
        plt.title(f"Original Distribution for {i}")
        plt.subplot(n, 3, j + 1)
        sns.histplot(x, bins=50, kde=True)
        plt.title(f"Quantile Transform for {i}")
        plt.subplot(n, 3, j + 2)
        sns.histplot(y, bins=50, kde=True)
        plt.title(f"Power Transform for {i}")

        # definimos j como los saltos de cada fin del bucle for para la siguiente fila
        # como en este caso son tres subplots por cada columna del dataframe, j añade 3 en cada vuelta
        j += 3
        # ya podemos usar nuestra función con el dataframe y nuestras columnas
        #test_transformers(df, columnas)

def report_plot(tree_entrenado, X_test, y_test, columnas_X):
    '''Funcion que toma de argumento el modelo y retorna el classification report y el gráfico del decission tree
    sirve para comparar distintos arboles, o modificaciones del mismo sin tener que ejecutar el bloque de código entero
    (ej. prunned tree de un DecisionTree ya creado).

    Para su ejecución se necesita el modelo ya entrenado (modelo.fit(X_train, y_train)) como argumento,
    por lo que la división en train y test también deberá darse previa al llamado de la función.

    Para más información sobre las métricas del Classification Report visita la documentación de la librería sci-kit learn al respecto.
    '''

    model_preds = tree_entrenado.predict(X_test)
    print(classification_report(y_test, model_preds))
    print('\n')
    plt.figure(figsize=(12, 8), dpi=150)
    plot_tree(tree_entrenado, filled=True, feature_names=columnas_X);

def bar_plot(df, columna):
    '''Ideal para columnas categoricas con no mas de 10 categorías.

    Esta función devuelve, con una sola línea de código: un plot dentro de un figsize=(15,6),
    un gráfico de barras con los colores de una paleta preseleccionada de colores, la frecuencia de dichos valores
    dentro del dataset, el título del gráfico, una etiqueta para el eje y, e imprime en pantalla un value counts
    de la columna que le hemos pasado a la función.

    En caso de querer usar la función con más de una columna, se recomienda usarla en un bucle for de la siguiente manera:

    for x in ['Col1', 'Col2', 'Col3', ...]:
        bar_plot(x)

        >> el output sería lo descrito anteriormente mostrado de forma consecutiva para cada columna.

    Si se requiere más información, visita la documentación de matplotlib en
    https://matplotlib.org/stable/index.html

    Agradecimientos a Enes Besinci,
    visita su perfil en kaggle en https://www.kaggle.com/code/enesbeinci/k-means-ile-m-teri-ki-ilik-analizi
    '''

    var = df[columna]
    varV = var.value_counts()
    plt.figure(figsize=(15, 6))
    plt.bar(varV.index, varV, color=mcolors.TABLEAU_COLORS)
    plt.xticks(varV.index)
    plt.ylabel("Frecuencia")
    plt.title(columna)
    plt.show()
    print("{}: \n {}".format(columna, varV))

def mapa_folium(df, geojson, key, coord, legend="Mapa"):
    """Esta función se apoya en la librería Folium para visualizar la distribución geográfica de los datos
    presentes en un dataset. Habrá que indicarle las coordenadas donde se inizializa el mapa, así como un archivo
    Json que almacene las geometrías GeoJson de las localidades del mapa.

    Se requiere la instalación de las librerías Pandas y Folium.

    --> df: dataframe que pasaremos a la función y que queremos visualizar. Deberá constar de dos columnas,
    la primera con los lugares del mapa que queremos mostrar y la segunda con los datos que queremos visualizar.
    --> geojson: archivo Json con las geometrías de las localidades que queremos mostrar.
    --> key: string, variable del Json geojson con la que vincularemos los datos. Deberá comenzar siempre por "feature"
    y estar escrita en nomenclatura javascript como por ejemplo "feature.id" o "featura.properties.statename".
    --> coord: tupla, coordenadas en [ejeX, ejeY] en las que queremos inicializar el mapa.
    --> legend: string, argumento opcional que indicará el título de la leyenda que siguen los datos mostrados en el mapa.
    """

    mapa = folium.Map(location=coord, zoom_start=4)

    tiles = ['stamenwatercolor', 'cartodbpositron', 'openstreetmap', 'stamenterrain']
    for tile in tiles:
        folium.TileLayer(tile).add_to(mapa)

    mapa.choropleth(
        geo_data=geojson,
        name='choropleth',
        data=df,
        columns=df.keys(),
        key_on=key,
        fill_color='BuPu',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=legend
    )
    folium.LayerControl().add_to(mapa)

    return mapa

def vis_line(df, ejex, ejey, group="", type=0):
    """Esta función nos permite visualizar los datos contenidos en un dataset mediante un line chart. Habrá que indicarle
    que columnas serán los ejes x e y del gráfico, así como indicarle por qué columna queremos agrupar los datos en caso que queramos,
    además también de poder elegir el tipo de line chart que queremos usar.

    Se requiere la instalación las librerías Plotly y Pandas.

    --> df: este será el dataframe que pasaremos a la función con la información que queremos visualizar en el gráfico.
    --> ejex: string, indicaremos la columna que queremos usar en el gráfico como eje X para representar los datos.
    --> ejey: string, indicaremos la columna que queremos usar en el gráfico como eje Y para representar los datos.
    --> group: string, es un argumento opcional que nos permite deicidir si queremos agrupar los datos o no. En caso de querer
    agruparlos tendremos que indicar el nombre de la columna por la cual queramos agruparlos.
    --> type: int, argumento opcional que nos permite decidir que tipo de line chart queremos utilizar, 0 para líneas y marcadores,
    1 para solo líneas y 2 u otro número para solo marcadores. Por defecto está activada la opción 0 para líneas y marcadores.
    """
    fig = go.Figure()
    if group != "":
        grupos = df[group].unique()
        if type==0:
            for grupo in grupos:
                x = df[df[group].values == grupo][ejex]
                y = df[df[group].values == grupo][ejey]
                fig.add_trace(go.Scatter(x=x, y=y,
                mode="lines+markers",
                name=grupo
                    ))
        elif type==1:
            for grupo in grupos:
                x = df[df[group].values == grupo][ejex]
                y = df[df[group].values == grupo][ejey]
                fig.add_trace(go.Scatter(x=x, y=y,
                mode="lines",
                name=grupo
                    ))
        else:
            for grupo in grupos:
                x = df[df[group].values == grupo][ejex]
                y = df[df[group].values == grupo][ejey]
                fig.add_trace(go.Scatter(x=x, y=y,
                mode="markers",
                name=grupo
                    ))

    else:
        if type == 0:
            fig.add_trace(go.Scatter(x=df[ejex], y=df[ejey],
                                     mode="lines+markers"))
        elif type == 1:
            fig.add_trace(go.Scatter(x=df[ejex], y=df[ejey],
                                     mode="lines"))
        else:
            fig.add_trace(go.Scatter(x=df[ejex], y=df[ejey],
                                     mode="markers"))
    return fig

def matrix_sca (df, dimensiones, agrupar, titulo="Scatter Matrix"):
    """Esta función nos permite crear una matriz de Scatterplots para realizar una comparativa de los valores de las
    variables de un dataset utilizando la librería Pyplot.express.

    Se requiere la instalación de las librerías Pyplot y Pandas

    --> df: dataframe que pasaremos a la función con la información que queremos comparar
    --> dimensiones: lista con los nombres de las columnas con cuyos datos queremos hacer una comparativa
    --> agrupar: string, nombre de la columna con la cual queremos agrupar los datos
    --> titulo: string, argumento opcional con el título que queremos poner a la gráfica. Por defecto será 'Scatter Matrix'
    """
    fig = px.scatter_matrix(df,
        dimensions=dimensiones,
        color=agrupar,
        title=titulo,
        labels=dimensiones)
    fig.update_traces(diagonal_visible=False)

    return fig