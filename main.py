#Arrays y matrices
import numpy as np

#Estructuras
import pandas as pd

#Visualización
import matplotlib.pyplot as plt
import plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot

#Preprocesamiento
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.compose import make_column_transformer

#Split
from sklearn.model_selection import train_test_split

#Feature Selection
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA

#Modelos
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression

#Metricas
from sklearn.metrics import silhouette_score, confusion_matrix, accuracy_score, mean_squared_error

#Ensembles
from sklearn.ensemble import IsolationForest, RandomForestRegressor, GradientBoostingClassifier, RandomForestClassifier

#DeepLearning
from sklearn.neural_network import MLPRegressor

#Pipelines
from sklearn.pipeline import make_pipeline

#Otros
import concurrent.futures as cf
import functools

def pca_visualization(df):

    """Escala los datos del dataframe original y aplica un PCA para reducir el numero de columnas original a 3 columnas.
    Teniendo 3 columnas / componentes principales, los datos pueden ser representados en un grafico de dispersión 3D.

    Args:
        df (DataFrame): Base de datos original sin la columna target

    Returns:
        3D Scatter Plot: Gráfico de dispersión 3D tras aplicar el PCA
        Varianza explicada acumulada para 3 componentes principales: Devuelve la suma de la varianza 
        explicada para 3 componentes principales. Esto informa de cuánta información se está perdiendo con el PCA.
    """
    
    scal = StandardScaler() 
    X_scal = scal.fit_transform(df)

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scal)

    principalDf = pd.DataFrame(data = X_pca, columns= ['P1', 'P2', 'P3'])

    xdata = principalDf['P1']
    ydata = principalDf['P2']
    zdata = principalDf['P3']

    trace1 = go.Scatter3d(x = xdata,
                        y = ydata,
                        z = zdata,
                        mode = 'markers',
                        marker = dict(size = 5, color = 'rgb(255,0,0)'))

    data = [trace1]
    layout = go.Layout(margin = dict(l=0,
                                    r=0,
                                    b=0,
                                    t=0))

    fig = go.Figure(data=data, layout=layout)

    expl = pca.explained_variance_ratio_
    print('Cumulative explained variance with 3 principal components:', round(np.sum(expl[0:3]),2))
    iplot(fig)

def my_pca(n_components, df):
    
    """Escala los datos del DataFrame original y aplica un PCA con el numero de componentes principales deseado.
    
        Args:
            df (DataFrame): Base de datos original sin la columna target
            
        Returns:
            Varianza explicada acumulada : Devuelve la suma de la varianza explicada para los componentes principales deseados. 
    """

    scal = StandardScaler() 
    X_scal = scal.fit_transform(df)

    pca = PCA(n_components = n_components)
    X_pca = pca.fit_transform(X_scal)

    expl = pca.explained_variance_ratio_
    print('Varianza acumulada', np.cumsum(expl))

    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')    
    plt.xticks(np.arange(0, n_components, step=1))

def my_kmeans(n_clusters, df):
    
    """Escala los datos del DataFrame original y aplica un KMeans con el numero de clusters deseado.
    
        Args:
            df (DataFrame): Base de datos original sin la columna target

        Returns:
            Inercias para cada modelo de KMeans y un grafico para visualizar los resultados
            Silhouette scores para cada modelo de Kmeans y un grafico para visualizar los resultados
         
    """
    
    scal = StandardScaler() 
    X_scal = scal.fit_transform(df)

    kmeans_per_k = [KMeans(n_clusters=k,random_state=42).fit(X_scal) for k in range(2, n_clusters+1)]
    inertias = [model.inertia_ for model in kmeans_per_k]   

    plt.figure(figsize=(8, 3.5))

    plt.plot(range(2, n_clusters+1), inertias, "bo-")
    plt.xlabel("$k$", fontsize=14)
    plt.ylabel("Inertia", fontsize=14)
    plt.show()

    silhouette_scores = [silhouette_score(X_scal, model.labels_) for model in kmeans_per_k]
    plt.figure(figsize=(8, 3))
    plt.plot(range(2, n_clusters+1), silhouette_scores, "bo-")
    plt.xlabel("$k$", fontsize=14)
    plt.ylabel("Silhouette score", fontsize=14)

    return inertias, silhouette_scores

################################### FUNCIÓN DE DETECCIÓN DE ANOMALÍAS ################################################
def anomalias_var(feature):
    '''Función para detectar las anomalías/outliers de las distintas variables (columnas/features) 
       de un dataset mediante un modelo de Isolation Forest.
        
       Sólo tiene un argumento de entrada:
         feature --> Ha de ser un VECTOR COLUMNA. Ejemplo: feature = df[["nombre_col"]]
         
       La función te devuelve un dataframe con los valores de la variable que el modelo considera anómalos'''
       
    model = IsolationForest(max_samples=len(feature), random_state=13)
    anomalias = model.fit_predict(feature)

    return feature[anomalias==-1]


#################################### FUNCIÓN DE FEATURE IMPORTANCE ###################################################
def FeatureImportance_rf(X,y,n):
    '''Función para plasmar el Feature Importance (ordenado de mayor a menor) de un dataset
       con valores numéricos, haciendo uso de un Random Forest Regressor.
    
       Los argumentos de entrada son:
            X --> el conjunto de variables (features) de tu dataset. Ha de tener formato dataframe o un np.array 2D
            y --> el target; con formato pd.Series o np.array 1D
            n --> nº de estimadores de los que quieres dotar al modelo Random Forest
            
        La función te devolverá un dataframe con 2 columnas:
            Relevancia --> indica el % de transcendencia que tiene una variable en relación al target.
            Variable --> indica el nombre de la columna (string)
    '''
    nombres = X.columns

    rf = RandomForestRegressor(n_estimators=n, random_state=13)
    rf.fit(X,y)

    decimales = rf.feature_importances_
    decimales = sorted(decimales, reverse=True)

    puntuacion = zip(map(lambda f:"{:.2%}".format(f),decimales), nombres)

    return pd.DataFrame(puntuacion, columns=['Relevancia', 'Variable'])


######################################## FUNCIÓN DE FEATURE SELECTION ##########################################
def FeatureSelection_var(X,min_var):

    '''Función para elegir el nº de variables adecuado para la cantidad de observaciones de un dataset
       según el filtro de varianza.
       
       Válida para aprendizaje NO SUPERVISADO (no hace falta el target).
       Ideal para datasets con pocas observaciones (filas) y muchas variables (columnas)
       
       La función primero te calcula el nº de variables adecuado para las filas halladas siguiendo la
       "rule of thumb";seguidamente, te aplica un preprocesado en el que te estandariza tus datos y
       y realiza un filtro de varianza (VarianceThreshold), por el que, por debajo de un mínimo valor de 
       varianza te desecha las variables que no lo cumplan.
       
       La función te devuelve un nuevo conjunto de datos (dataset sin target) estandarizado y únicamente con
       aquellas columnas que superen el valor mínimo de varianza, siempre que éstas sean una cantidad igual o
       inferior al número adecuado de variables indicado por la "rule of thumb". En caso de que todas las 
       variables tuvieran una varianza superior al valor mínimo y éstas fueran demasiadas, la función iría
       subiendo automáticamente ese umbral mínimo hasta que se redujeran las columnas.

       Como argumentos de entrada se tienen:
            X --> dataset (sin target) de valores numéricos
            min_var --> valor mínimo de varianza que deseamos que tengan las variables del dataset
       
       '''
    f=len(X)
    N = 5*np.log10(f)

    var_pipe = make_pipeline(StandardScaler(), VarianceThreshold(min_var))
    selector = var_pipe.fit(X)
    variables = selector.get_feature_names_out()
    pipelado = var_pipe.transform(X)
    columnas = pipelado.shape[1]
    
    while columnas > N:
        min_var += 0.1
        var_pipe = make_pipeline(StandardScaler(), VarianceThreshold(min_var))
        selector = var_pipe.fit(X)
        variables = selector.get_feature_names_out()
        pipelado = var_pipe.transform(X)
        columnas = pipelado.shape[1]

        if min_var == 1.0:
            break
    
    X_new = pd.DataFrame(pipelado, columns=list(variables))
    return X_new

def Impute_Tree_Regressor(df: pd.core.frame.DataFrame, n_max_depth: int, random_state: int) -> pd.core.frame.DataFrame:
    """
    Método de imputación de variables contínuas, a través de un árbol de decisión.
    
    EXCLUSIVAMENTE ÚTIL PARA MATRICES NUMÉRICAS, ONE HOT ENCODER/ LABEL ENCODER HA DE ESTAR YA REALIZADO

    cuando se dispone exclusivamente de un NaN por fila, en caso de que haya más de un NaN por fila se eliminará la fila del df.
    """

    # Quitamos los warnings para un output más limpio del código
    import sys

    if not sys.warnoptions:
        import warnings
        warnings.simplefilter("ignore")

    # eliminamos las filas con Nan>1 como se explica arriba    
    df = df.dropna(thresh = df.shape[1]-1)
    df2 = df.dropna()

    # Comienza el bucle
    while (sum(df.isnull().any(1))>0):
        # identificar fila y columna del NAN
        nan_rows = df[df.isnull().any(1)]
        nan_columns = df.columns[df.isnull().any()]
        for i in nan_columns:
            for j in nan_rows.index:
                # En caso de que sea Nan
                if (np.isnan(df.loc[j,i])):
                    # separo el target de entrenamiento
                    target = df2[i]
                    # separo mi X_train
                    X_train = df2.drop(i, axis=1)
                    # defino y entreno el modelo
                    tree_reg = DecisionTreeRegressor(max_depth = 2, random_state=42)
                    tree_reg.fit(X_train,target)
                    # obtengo la fila a la que se imputa el Nan
                    X_test = df.loc[[j]]
                    X_test = pd.DataFrame(X_test)
                    X_test.drop(i, axis=1, inplace=True)
                    # realizo su predicción
                    pred = tree_reg.predict(X_test)
                    # sustituyo en df
                    df.loc[j,i] = pred
                    # actualizo las condiciones 
                    nan_rows = df[df.isnull().any(1)]
                    nan_columns = df.columns[df.isnull().any()]
                else:
                    continue
    return df

def Impute_Tree_classifier(df:  pd.core.frame.DataFrame, categorical_variable: str, n_max_depth: int, random_state:int) -> pd.core.frame.DataFrame:
    """
    Método de imputación de variables categóricas, a través de un árbol de decisión.
    
    EXCLUSIVAMENTE ÚTIL PARA MATRICES NUMÉRICAS, ONE HOT ENCODER/ LABEL ENCODER HA DE ESTAR YA REALIZADO

    Aplicable cuando se dispone exclusivamente de un NaN por fila, en caso de que haya más de un NaN por fila se eliminará la fila del df
    """

    # Quitamos los warnings para un output más limpio del código
    import sys

    if not sys.warnoptions:
        import warnings
        warnings.simplefilter("ignore")

    # eliminamos las filas con Nan>1 como se explica arriba    
    df = df.dropna(thresh = df.shape[1]-1)
    
    new_df = df[[categorical_variable]]
    df2 = df.dropna()

    # Comienza el bucle
    while (sum(new_df.isnull().any(1))>0):
        # identificar fila y columna del NAN
        nan_rows = new_df[new_df.isnull().any(1)]
        for j in nan_rows.index:
            # En caso de que sea Nan
            if (np.isnan(new_df.loc[j])[0]):
                # separo el target de entrenamiento
                target = df2[categorical_variable]
                # separo mi X_train
                X_train = df2.drop(categorical_variable, axis=1)
                # defino y entreno el modelo
                tree_clf = DecisionTreeClassifier(max_depth = n_max_depth, random_state=random_state)
                tree_clf.fit(X_train,target)
                # obtengo la fila a la que se imputa el Nan
                X_test = df.loc[[j]]
                X_test = pd.DataFrame(X_test)
                X_test.drop(categorical_variable, axis=1, inplace=True)
                # realizo su predicción
                pred = tree_clf.predict(X_test)
                #sustituyo en df
                df.loc[j,categorical_variable] = pred
                # actualizo las condiciones
                new_df.loc[j,categorical_variable] = pred
                nan_rows = df[df.isnull().any(1)]
            else:
                continue
    return df

def relative_absolute_error(y_train: pd.core.series.Series, y_test: pd.core.series.Series, y_predicted: pd.core.series.Series, type_metric = 'error') -> float:
    """ 
    Función que calcula la métrica de error Relative Absolute error o bien su score
    si así se desea ha de cambiarse type_metric por score.

    El numerador se compone por la suma de los errores absolutos,
    el denominador se compone por el error de un modelo trivial, que en este caso consiste en la media del target train.

    Esta función calcula una metrica que por lo general estará acotada entre abs(0,1) siempre y cuando nuestro modelo prediga mejor que el modelo trivial.
    """
    # En caso que las longitudes de los vectores no sean iguales dar un error
    if ((len(y_test)==len(y_predicted)) == False):
        raise TypeError ("y_test and y_predicted han de ser de la misma longitud")
    else:
        # En caso que en type_metric se introduzca otro string diferente a los permitidos dar un error
        if (type_metric) not in (['error', 'score']):
            raise TypeError ("type_metric ha de ser error (default) o score")
        # calculo de la métrica
        else:
            if (type_metric == 'score'): 
                numerator = np.sum(np.abs(y_predicted-y_test))
                denominator = np.sum(np.mean(y_train)-y_test)
                return (-(numerator/denominator))
            else:
                numerator = np.sum(np.abs(y_predicted-y_test))
                denominator = np.sum(np.mean(y_train)-y_test)
                return (numerator/denominator)

def specificity(y_true,y_pred):
    '''esta función determina la especificidad utilizando una matriz de confusión.
    Args:
        y_true: el verdadero target
        y_pred: el target previsto

    return:
        valor de especificidad

    '''
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn+fp)
    return specificity

def classifier_cat(dataf):
    ''' 
        Descripcion: Funcion para clasificar las varibles categoricas

        Args: dataframe 
        
        Returns: dos listas. Una lista con las variables binarias y otra con las NO binarias
    '''
    categorias = dataf.columns[dataf.dtypes == 'object']
    # Si nuestras columnas categoricas tienen sólo dos valores, utilizar Label Encoder sino One Hot Encoder

    categorias_bin = []
    categorias_NO_bin = []

    for i in categorias:
        if dataf[i].nunique() <= 2:
            categorias_bin.append(i)
        else:
            categorias_NO_bin.append(i)

    return categorias_bin, categorias_NO_bin

def cat_to_num(dataf):
    ''' 
        Descripcion: Funcion para transformar valores categoricos a numericos,
                    teniendo en cuenta la cantidad de valores en cada columna,
                    usando Label Encoder y One Hot Encoder

        Args: dataframe 
        
        Returns: dataframe con los valores categoricos transformados en numericos
    '''
    
    # Si nuestras columnas categoricas tienen sólo dos valores, utilizar Label Encoder sino One Hot Encoder
    le = LabelEncoder()
    ohe = OneHotEncoder(handle_unknown='ignore')

    bin, NO_bin = classifier_cat(dataf)
    
    for i in bin:
        #print(i)
        le.fit(dataf[i])
        dataf[i] = le.transform(dataf[i])

    # verbose_feature_names_out=False es para mantener los nombres sin prefijos
    transformer = make_column_transformer((ohe, NO_bin), remainder='passthrough', verbose_feature_names_out=False)
    
    transformer.fit(dataf)
    transformed = transformer.transform(dataf)
    

    #transformed = transformer.fit_transform(df)

    dataf = pd.DataFrame(transformed, columns=transformer.get_feature_names_out())
            

    return dataf

# Ver balance de datos del target
def ver_balance(target):
    ''' 
        Descripcion: Funcion para ver el balance de los datos del target.

        Args: dataframe['target'] 
        
        Returns: dataframe balanceado
    '''
    # Vemos si mis datos estan balanceados
    print(target.value_counts())
    ax = target.value_counts().plot(kind = 'bar')
    ax.set_title('0 = no tiene, 1 = sí tiene')
    ax.set_ylabel('Cantidad')
    plt.show()
    return ax

def under(X, y):
    ''' 
        Descripcion: Funcion que balancea los dataframes con un undersampling.

        Args: X_dataframe, y_dataframe
        
        Returns: X_dataframe e y_dataframe balanceados
    '''
    subsample = RandomUnderSampler(random_state = 42)

    X_train_new, y_train_new = subsample.fit_resample(X, y)

    return X_train_new, y_train_new

def over(X, y):
    ''' 
        Descripcion: Funcion que balancea los dataframes con un oversampling.

        Args: X_dataframe, y_dataframe
        
        Returns: X_dataframe e y_dataframe balanceados
    '''
    oversample = SMOTE(random_state=42)
    X_train_new, y_train_new = oversample.fit_resample(X, y)
   
    return X_train_new, y_train_new

def sampling(X, y):
    ''' 
        Descripcion: Funcion que dependiendo de la diferencia entre 0s y 1s en y_dataframe,
                     balancea los dataframes(X_dataframe e y_dataframe) con un undersampling,
                     oversampling, oversampling-undersampling o los deja igual.

        Args: X_dataframe, y_dataframe
        
        Returns: X_dataframe  e y_dataframe balanceados
    '''

    # Contar cuantos valores de cada tipo en el target tenemos
    target_0 = y.value_counts()[0]
    print('Cantidad de 0s inicial: ', target_0)
    target_1 = y.value_counts()[1]
    print('Cantidad de 1s inicial: ', target_1)
  
    # Comprobamos la diferencia entre el numero de 0s y 1s
    if (abs(target_0 - target_1) > 150) and ( abs(target_0 - target_1) <= 2000):
        # Hacemos oversampling para ajustar el dataset
        X_new, y_new = over(X, y)

        target_0 = y_new.value_counts()[0]
        print('Cantidad de 0s final: ', target_0)
        target_1 = y_new.value_counts()[1]
        print('Cantidad de 1s final: ', target_1)
        return X_new, y_new
    # Comprobamos la diferencia entre el numero de 0s y 1s
    elif (abs(target_0 - target_1) > 2000) and ( abs(target_0 - target_1) <= 5000):
        # Hacemos oversampling-undersampling para ajustar el dataset
        X_n, y_n = over(X, y)
        X_new, y_new = under(X_n, y_n)

        target_0 = y_new.value_counts()[0]
        print('Cantidad de 0s final: ', target_0)
        target_1 = y_new.value_counts()[1]
        print('Cantidad de 1s final: ', target_1)
        return X_new, y_new
    # Comprobamos la diferencia entre el numero de 0s y 1s
    elif (abs(target_0 - target_1) > 10000):
        # Hacemos undersampling para ajustar el dataset
        X_new, y_new = under(X, y)

        target_0 = y_new.value_counts()[0]
        print('Cantidad de 0s final: ', target_0)
        target_1 = y_new.value_counts()[1]
        print('Cantidad de 1s final: ', target_1)
        return X_new, y_new
    else:
        # Dejamos los dataframes tal cual
        X_new, y_new = X, y
        target_0 = y_new.value_counts()[0]
        print('Cantidad de 0s final: ', target_0)
        target_1 = y_new.value_counts()[1]
        print('Cantidad de 1s final: ', target_1)
        return X_new, y_new

def gradBoosting(X_train, X_test, y_train, y_test):
    ''' 
        Descripcion: Funcion que encuentra el mejor score y su estimador usando el modelo Gradient Boosting
                     Calcula uno a uno los scores.

        Args: X_train, X_test, y_train, y_test
        
        Returns: best_estimator, max_score
    '''
    score_max = 0
    cont = 0
    estimator_param = np.arange(50,2000,15)
    best_estimator = 0

    print(estimator_param)
    
    for i in estimator_param:
        gbrt_clf = GradientBoostingClassifier(criterion='friedman_mse', random_state = 42, n_estimators = i, learning_rate=0.1)                   
        gbrt_clf.fit(X_train, y_train)

        goal = gbrt_clf.score(X_test, y_test)
        

        
        if score_max < goal:
            score_max = goal
            best_estimator = i
        else:
            # Si el score no mejore en 5 estimadores, salgo del bucle
            if cont < 5:
                score_max = score_max
                cont += 1
                continue
            else:
                break
    
        print(best_estimator,score_max)

    return best_estimator, score_max

# here I check the unique varible
def cal_cols(df,column,n=0):
    '''df: Dataframe,
    columns: columnas a realizar encode
    revisa el numero de columnas necesarias para el encode
    basado en el numero de categorias de las columnas'''
    num_val=2**n
    if num_val>=df[column].nunique():
        return n
    else:
        n+=1
        return cal_cols(df,column,n)

# here I do an array with all posible binary combination
def bi_ray(n,bi=[[1],[0]],num_loop=1):

    ''' n: numero de columnas necesarias
    Crea un array con todas las columnas binarias posibles en n columnas'''

    if n==1:
        return np.array(bi)
    else:
        list_bi=list(map(lambda x: x+[0] if x[-1]==[1] else x+[1],bi))
        list_bi1=list(map(lambda x: x+[1] if x[-1]==[1] else x+[0],bi))
        bi=list_bi+list_bi1
    if num_loop+1==n:
        return np.array(bi)
    else:
        num_loop+=1
        return bi_ray(n,bi,num_loop)

# here I transform that array to a frame
def frame_maker(df,columns,up_array,num_loops=0,num_col=0,dict_decod={},full_frame=pd.DataFrame([])):
    '''df: Dataframe
    columns: lista de columnas con todas las columnas a codificar
    up_array: array utilizado para el encode
    Crea un data frame con todas las columnas que se ha decidido codificar'''
    
    frame=pd.DataFrame(df[columns[num_loops]].unique())
    
    
    frame[columns[num_loops]+"_"+str(num_col)]=up_array[num_loops][:,num_col]
    
    frame=frame.rename(columns={0:columns[num_loops]})
    
    df=pd.merge(df, frame, on=[columns[num_loops]])
    
    if len(up_array[num_loops][0])!=num_col+1:
        num_col+=1
        
        if num_col==1:
            
            full_frame=pd.DataFrame(df[columns[num_loops]].unique()).rename(columns={0:columns[num_loops]})
            
            full_frame=pd.merge(full_frame,frame,on=columns[num_loops])
        else:
            
            full_frame=pd.merge(full_frame,frame,on=columns[num_loops])
        return frame_maker(df,columns,up_array,num_loops,num_col,dict_decod,full_frame)
    else:
        if num_col==0:
            full_frame=frame
            dict_decod[columns[num_loops]]=full_frame
            full_frame=pd.DataFrame([])
            df=df.drop(columns[num_loops],axis=1)

        else:
            num_col=0

            full_frame=pd.merge(full_frame,frame,on=columns[num_loops])
            dict_decod[columns[num_loops]]=full_frame
            full_frame=pd.DataFrame([])
            df=df.drop(columns[num_loops],axis=1)
    
    if len(columns)!=num_loops+1:
        
        num_loops+=1
        
        return frame_maker(df,columns,up_array,num_loops,num_col,dict_decod,full_frame)
    return df,dict_decod



def bi_hot_encoding(df,columns=None):

    '''df: DataFrame
    columns: lista de columnas con todas las columnas a codificar, default None todas las columnas serán codificadas
    Crea un dataframe codificado añadiendo al nombre de las columnas codificadas + "_number",
    y un diccionario como claves el nombre de la columna codificada y valores un dataframe con los valores codificados'''
    
    if columns==None:
        columns=list(df.select_dtypes("object").columns)

    
    n=list(map(lambda x: cal_cols(df,x,n=0), columns))

    list_array=list(map(lambda x: bi_ray(n=x), n))

    num_feat=list(map(lambda x: df[x].nunique() ,columns))

    up_array=list(map(lambda x: list_array[num_feat.index(x)][:x] , num_feat))

    df=frame_maker(df,columns,up_array)

    return df

def get_clusters(X_train,cluster_fd=KNeighborsClassifier(),cluster_mk=DBSCAN()):

    """esta función recoge los datos de entrenamiento y define clusters no supervisados mediante DBSCAN,
     a continuación entrena un modelo KNeighborsClassifier para predecir los clusters en los datos de prueba. 
    Args:
        X_train: los datos de entrenamiento.

    Returns:
        los datos de entrenamiento con el cluster colum y un modelo KNeighborsClassifier entrenado.
    """
    

    predictions=cluster_mk.fit_predict(X_train)

    cluster_fd.fit(X_train,predictions)

    if str(type(X_train))=="<class 'pandas.core.frame.DataFrame'>":
        
        X_train=pd.concat([X_train,pd.DataFrame(predictions).rename(columns={0:"cluster"})],axis=1)
    
    else:

        X_train=pd.concat([pd.DataFrame(X_train),pd.DataFrame(predictions).rename(columns={0:"cluster"})],axis=1)
    
    return X_train,cluster_fd

def model_dic(df_model,n=0,dic={}):

    """esta función toma una lista de modelos entrenados, los datos de entrenamiento y su numero de cluster, 
    y hace un diccionario con el número de cluster como clave y la lista como valor.
    Args:
        df_model: lista de modelos entrenados, los datos de entrenamiento y su numero de cluster.

    Returns:
        diccionario con el número de cluster como clave y la lista como valor.
    """
    dic.update({df_model[n][3]:df_model[n]})

    n+=1

    if len(df_model)==n:
        return dic

    return model_dic(df_model,n,dic)


class cluster_ensemble:

    """
    este es el objeto cluster_ensemble, 
    hace predicciones de datos basadas en los clusters que identifica usando DBSCAN
    
    Atributos:
        
        model: el modelo que se utilizará para realizar las predicciones. 
        por defecto: RandomForestClassifier
        
        cluster_fd: el modelo que predice los clusters utilizando las evaluaciones de DBSCAN.
        por defecto: KNeighborsClassifier 

        cluster_mk: el modelo DBSCAN básico que calificará los clusters en el proceso de entrenamiento
        por defecto: DBSCAN, con los parámetros por defecto, no puede ser otro modelo

        dic_model: diccionario de modelos vacío por defecto

    Metodos:

        fit:
            este método entrena el modelo en los diferentes clusters en los datos de entrenamiento,
            al final de este proceso tenemos un diccionario de modelos entrenados y
            un modelo para clasificar los clusters en los datos de test.

            Args:
                X_train: datos de entrenamiento
                y_train: target de entrenamiento
                self: todos los atributos
            
            Returns:
                    entrena todos os atributos
            

        predict:
            este método predice el target

            Args:
                X_test: datos de test
                self: self.cluster_fd, self.dic_model
            
            Returns:
                    predice el target
            
    
    """
    
    def __init__(self,model=RandomForestClassifier(),cluster_fd=KNeighborsClassifier(),cluster_mk=DBSCAN(),dic_model={}):
        self.model=model
        self.cluster_fd=cluster_fd
        self.cluster_mk=cluster_mk
        self.dic_model=dic_model

    def fit(self,X_train,y_train):

        X_train,self.cluster_fd=get_clusters(X_train,self.cluster_fd,self.cluster_mk)

        if str(type(y_train))=="<class 'pandas.core.frame.DataFrame'>":
            
            X_train=pd.concat([X_train,y_train],axis=1)
        
        else:

            X_train=pd.concat([X_train,pd.DataFrame(y_train)],axis=1)
        
        with cf.ThreadPoolExecutor() as excutor:
            
            df_list=list(excutor.map(lambda x: [np.array(X_train[X_train["cluster"]==x].iloc[:,:-2]),np.array(X_train[X_train["cluster"]==x].iloc[:,-1]),x],X_train["cluster"].unique()))
            
        
        df_model=list(map(lambda x:[self.model.fit(x[0],x[1])]+x,df_list))
            
        
        self.dic_model=model_dic(df_model)

    


    def predict(self,X_test):

        if str(type(X_test))=="<class 'pandas.core.frame.DataFrame'>":

            X_test["cluster"]=self.cluster_fd.predict(X_test)

        else:
            
            X_test=pd.DataFrame(X_test)

            X_test["cluster"]=self.cluster_fd.predict(X_test)
        
        preds=list(map(lambda x: self.dic_model[x][0].predict(X_test[X_test["cluster"]==x].iloc[:,:-1]).tolist()[0], self.dic_model.keys()))

        preds=list(functools.reduce(lambda x, y:x+y,preds))

        preds_val=preds[::2]
        preds_index=preds[1::2]

        preds_val=list(functools.reduce(lambda x, y:x+y,preds_val))
        preds_index=list(functools.reduce(lambda x, y:x+y,preds_index))

        preds=list(map(lambda x:preds_val[x] ,preds_index))


        return preds

def Dec_tree_clf(X,y):
    '''
    Función para obtener, con bases de datos numéricas y limpias, 10 valores de score,
    en un Árbol Clasificador, con parámetros random (que también se obtendrán).

    X -> Dataframe sin nuestro target
    y -> Target

    return -> dataframe con parámetros ordenados por score
    '''
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.25,
                                                    random_state=42)
    
    max = X.shape[1]+1
    
    # En cada lista irán los valores de un parámetro
    params1 = []
    params2 = []
    params3 = []

    # En estas listas irán los valores de accuracy de train y test 
    Pred_train_accuracy = []
    Pred_test_accuracy = []
    
    # El objetivo del bucle while es probar 100 combinaciones distintas de parámetros.
    # La función mostrará los 10 mejores accuracy obtenidos en test, así como los parámetros necesarios.
    i = 0
    while i<100:
        param1 = np.arange(1,6).tolist()
        param2 = np.arange(2,12).tolist()
        param3 = np.arange(3,max).tolist()

        tree_clf = DecisionTreeClassifier(random_state = 42,
                                min_samples_leaf= np.random.choice(param1),
                                max_depth= np.random.choice(param2), 
                                max_features = np.random.choice(param3)
                                )
    
        tree_clf.fit(X_train,y_train)

        y_pred_train = tree_clf.predict(X_train)
        y_pred_test = tree_clf.predict(X_test)

        #Añadimos a cada lista los valores que le corresponden

        params1.append(str(tree_clf.get_params)[65:77])
        params2.append(str(tree_clf.get_params)[78:94])
        params3.append(str(tree_clf.get_params)[94:114])

        Pred_train_accuracy.append(accuracy_score(y_train, y_pred_train))

        Pred_test_accuracy.append(accuracy_score(y_test, y_pred_test))

        i+=1

    #Creamos un dataframe con las 5 listas
    df_1 = pd.DataFrame()
    
    df_1['Parámetro 1'] = params1
    df_1['Parámetro 2'] = params2
    df_1['Parámetro 3'] = params3
    df_1['Parámetro 4'] = 'random state = 42'
    df_1['accuracy en train'] = Pred_train_accuracy
    df_1['accuracy en test'] = Pred_test_accuracy
    
    return df_1.sort_values(by='accuracy en test',ascending=False).head(10)

def LogisticReg(X,y):
    '''
    Función para obtener, con bases de datos numéricas y limpias, 10 valores de score,
    en una Logistic Regression, con parámetros random (que también se obtendrán).

    X -> Dataframe sin nuestro target
    y -> Target

    return -> dataframe con parámetros ordenados por accuracy
    '''

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.25,
                                                    random_state=42)
    
    # En cada lista irán los valores de un parámetro
    params1 = []

    # En estas listas irán los valores de accuracy de train y test 
    Pred_train_accuracy = []
    Pred_test_accuracy = []
   
    # El objetivo del bucle while es probar 100 combinaciones distintas de parámetros.
    # La función mostrará los 10 mejores accuracy obtenidos en test, así como los parámetros necesarios.
    param1 = np.arange(0, 1000, 1).tolist()
    try:
        param2 = ['l1','l2', 'elasticnet','none']
        i =0

        while i<100:
            lr = LogisticRegression(penalty = np.random.choice(param2),
                                    C = np.random.choice(param1) )
        
            lr.fit(X_train,y_train)
        

            y_pred_train = lr.predict(X_train)
            y_pred_test = lr.predict(X_test)

            #Añadimos a cada lista los valores que le corresponden

            params1.append(str(lr.get_params)[60:])

            Pred_train_accuracy.append(accuracy_score(y_train, y_pred_train))

            Pred_test_accuracy.append(accuracy_score(y_test, y_pred_test))

            i+=1
    
    #PROBLEMA
    #En algunas ocasiones solo te acepta, como calor de penalty, l2 o none
    except ValueError:
        param2 = ['l2', 'none']
    
        i =0

        while i<100:
            lr = LogisticRegression(penalty = np.random.choice(param2),
                                    C = np.random.choice(param1) )
        
            lr.fit(X_train,y_train)
        

            y_pred_train = lr.predict(X_train)
            y_pred_test = lr.predict(X_test)

            #Añadimos a cada lista los valores que le corresponden

            params1.append(str(lr.get_params)[60:])

            Pred_train_accuracy.append(accuracy_score(y_train, y_pred_train))

            Pred_test_accuracy.append(accuracy_score(y_test, y_pred_test))

            i+=1

    #Creamos un dataframe con las 5 listas
    df_1 = pd.DataFrame()
    
    df_1['Parámetros'] = params1
    df_1['accuracy en train'] = Pred_train_accuracy
    df_1['accuracy en test'] = Pred_test_accuracy
    
    return df_1.sort_values(by='accuracy en test',ascending=False).head(10)

def RandomForest(X,y):
    '''
    Función para obtener, con bases de datos numéricas y limpias, 10 valores de score,
    en un Random Forest, con parámetros random (que también se obtendrán)

    X -> Dataframe sin nuestro target
    y -> Target

    return -> dataframe con parámetros ordenados por accuracy
    '''
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.25,
                                                    random_state=42)
    
    # En cada lista irán los valores de un parámetro
    params1 = []
    params2 = []
    params3 = []
    max = X.shape[1]+1

    # En estas listas irán los valores de accuracy de train y test 
    Pred_train_accuracy = []
    Pred_test_accuracy = []
    
    # El objetivo del bucle while es probar 100 combinaciones distintas de parámetros.
    # La función mostrará los 5 mejores accuracy obtenidos en test, así como los parámetros necesarios.
    i = 0
    while i<100:
        param1 = np.arange(1,200).tolist()
        param2 = np.arange(2,12).tolist()
        param3 = np.arange(3,max).tolist()

        rnd_clf = RandomForestClassifier(random_state = 42,
                                n_estimators = np.random.choice(param1),
                                max_depth= np.random.choice(param2), 
                                max_features = np.random.choice(param3))
    
        rnd_clf.fit(X_train,y_train)

        y_pred_train = rnd_clf.predict(X_train)
        y_pred_test = rnd_clf.predict(X_test)

        #Añadimos a cada lista los valores que le corresponden

        params1.append(str(rnd_clf.get_params)[65:77])
        params2.append(str(rnd_clf.get_params)[78:93])
        params3.append(str(rnd_clf.get_params)[94:110])

        Pred_train_accuracy.append(accuracy_score(y_train, y_pred_train))

        Pred_test_accuracy.append(accuracy_score(y_test, y_pred_test))

        i+=1

    #Creamos un dataframe con las 5 listas
    df_1 = pd.DataFrame()
    
    df_1['Parámetro 1'] = params1
    df_1['Parámetro 2'] = params2
    df_1['Parámetro 3'] = params3
    df_1['Parámetro 4'] = 'random state = 42'
    df_1['accuracy en train'] = Pred_train_accuracy
    df_1['accuracy en test'] = Pred_test_accuracy
    
    return df_1.sort_values(by='accuracy en test',ascending=False).head(10)

def pickleizer(nombre, modelo=None):
    ''' pickleizer tiene la capacidad de guardar modelos ya entrenados o abrir modelos entrenados desde una carpeta. 
        Tiene la ventaja de que la propia ejecución de la funcion importa la libreria pickle.

        Args: 
            nombre: Nombre del archivo donde se quiere importar o guardar el modelo.
            modelo: En caso de que se seleccione un modelo (ya entrenado) lo guardará. Cuando no se especifíca un modelo lo abre. Default = None'''
    import pickle
    if modelo== None:
        with open(nombre, 'rb') as archivo_entrada:
            trained_model= pickle.load(archivo_entrada)
        return(trained_model)
    else:
        with open(nombre, 'wb') as archivo_salida:
            pickle.dump(modelo, archivo_salida)


def DPRegressor(X: pd.DataFrame, y: pd.Series):
    '''
    Regresor con capas ocultas de neuronas
    Esta función solo se puede usar si todas las colunmnas son numericas

    Args:
    X -> DataFrame
        Matriz con los datos
    y -> Series
        Vector con el target

    Return:
    model -> Pipeline
        Modelo ya entrenado
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    mlp_reg = MLPRegressor(hidden_layer_sizes=[50, 50, 50], random_state=42)

    #Debemos escalar los datos ya que utiliza descenso de gradiente
    pipeline = make_pipeline(StandardScaler(), mlp_reg)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred, squared=False)

    print('El error medio es de:', rmse)

    return pipeline