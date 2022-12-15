#Arrays y matrices
import numpy as np

#Estructuras
import pandas as pd

#WebScrapping
from selenium import webdriver
from selenium.webdriver.common.by import By
import time

#Manejo de mi directorio personal
from multiprocessing.sharedctypes import Value
import requests
import io
from datetime import datetime as dt
from PIL import Image
import time
import os

#Librerias secundarias
import shutil
import cv2
from sklearn.utils import shuffle
from datetime import date
from unidecode import unidecode
from collections import Counter

def read_it(url):

    """Mediante la función read_it() buscamos cargar un dataset 
    sin tener que preocuparnos por el tipo de archivo que sea (con 
    diferentes separadores) ni la codificación que tenga, 

    Args:
      url (URL): Link o ubicación del dataset que vamos a trabajar

    Returns:
      Devuelve un dataset ya transformado para poder trabajar con pandas
      """ 

    sep=["\t",",",";","|",":"]
    encoding=["utf_32","utf_16","utf_8"]

    for i in sep:
        for j in encoding:
         
            try: 
             pd.read_csv(url, sep=i, encoding=j)

            except:
             continue
         
            if pd.read_csv(url, sep=i, encoding=j).shape[1]==1:
             continue
            
            else:
             return pd.read_csv(url, sep=i, encoding=j)


def google_img(path_api, urls, directory_names, directory_path):

    """Mediante google_img() buscamos automatizar la descarga de imagenes relacionadas
        con una busqueda determinada en nuestro navegador de internet, nos
        apoyaremos en la API de Selenium y simularemos una serie de busquedas 
        para posteriormente descargar dichas imagenes en las carpetas que previamente 
        hemos nombrado

        Args:
            path_api ("string"): Ubicación de la API del navegador web de Selenium 
            urls ("lista"): Lista de strings que componen las URLs de las imagenes que queremos obtener
            directory_names("lista"): Lista de strings con los nombres que queremos llamar a las carpetas donde estarán las imagenes (y las imagenes)
            directory_path("string"): Ubicación donde crear dichas carpetas

        Returns:
            Devuelve "n" carpetas con la mayor cantidad de imagenes posibles descargadas de la URLs que les hemos pasado
        """


    wd = webdriver.Chrome(executable_path=path_api)


    def get_images_from_google(wd, delay, max_images, url):
        def scroll_down(wd):
            wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(delay)

        url = url
        wd.get(url)

        image_urls = set()
        skips = 0
        while len(image_urls) + skips < max_images:
            scroll_down(wd)
            thumbnails = wd.find_elements(By.CLASS_NAME, "Q4LuWd")

            for img in thumbnails[len(image_urls) + skips:max_images]:
                try:
                    img.click()
                    time.sleep(delay)
                except:
                    continue

                images = wd.find_elements(By.CLASS_NAME, "n3VNCb")
                for image in images:
                    if image.get_attribute('src') in image_urls:
                        max_images += 1
                        skips += 1 
                        break

                    if image.get_attribute('src') and 'http' in image.get_attribute('src'):
                        image_urls.add(image.get_attribute('src'))

        return image_urls


    def download_image(down_path, url, file_name, image_type='JPEG',
                    verbose=True):
        try:
            time = dt.now()
            curr_time = time.strftime('%H:%M:%S')
            img_content = requests.get(url).content
            img_file = io.BytesIO(img_content)
            image = Image.open(img_file)
            file_pth = down_path + file_name

            with open(file_pth, 'wb') as file:
                image.save(file, image_type)

            if verbose == True:
                print(f'The image: {file_pth} downloaded successfully at {curr_time}.')
        except Exception as e:
            print(f'Unable to download image from Google Photos due to\n: {str(e)}')




    if __name__ == '__main__':
        google_urls = urls
        nombre_carpeta = directory_names

        coin_path = directory_path
    
        for lbl in nombre_carpeta:
            if not os.path.exists(coin_path + lbl):
                print(f'Making directory: {str(lbl)}')
                os.makedirs(coin_path+lbl)

        for url_current, lbl in zip(google_urls, nombre_carpeta):
            urls = get_images_from_google(wd, 0, 100, url_current)
        
            for i, url in enumerate(urls):
                download_image(down_path=coin_path+lbl+"/", 
                            url=url, 
                            file_name=str(i+1)+ '.jpg',
                            verbose=True) 
        wd.quit()

def dif_encoder(x,y):
    '''dif_encoder devuelve una nueva columna en el Dataframe codificando en 0 y 1 
        en base a la resta entre dos columnas seleccionadas por el usuario

            0 -> La diferencia es negativa
            1 -> La diferencia es positiva
        Args:
            x: Columna 1 del dataframe
            y: Columna 2 del dataframe
        Returns:
            Una nueva columna dentro del Dataframe con la codificación en 0 y 1

    '''

    return np.where(x>y,1,0)


def splityear(x):
    '''splityear devuelve el año de una fecha completa dd/mm/aaaa
        
        Args:
            x: Columna de dataframe que se quiere modificar
        
        Returns:
            Año que aparece en la fecha completa'''


    for i in x:
      año = [int(i.split('/')[2]) for i in x]
      return año


def simbolcleaner (x):
    '''simbolcleaner limpia aquellos elementos numéricos que contienen caracteres especiales
        Args:
            x: Columna de dataframe que se quiera limpiar de caracteres especiales
        Returns:
            Números sin caracteres especiales

    '''

    lista=[]
    for i in x:
        lista.append(''.join(filter(str.isalnum,i)))
    return lista



def too_many_nans(df, threshold=0, clean=True):

    '''too_many_nans permite eliminar las columnas seleccionadas en funcion del porcentaje de valores nulos que haya en ellas. 
        Args:
            df: El dataframe
            threshold: El porcentaje de valores nulos a partir del cual se quieren eliminar columnas. Default=0
            clean: Cuando es True se eliminan las columnas. Cuando es False se genera un nuevo df en el que se muestra el porcentaje de valores
                   nulos de cada columna. Default=True

    '''
    na = df.isna().sum() 
    n= len(df) 
    percent= np.round(na*100/n,2)
    df_perc= pd.DataFrame(percent, columns= ['Nans Percentage']).sort_values('Nans Percentage', ascending=False)
    if clean == False:
        return df_perc

    elif clean== True:
        cols_to_drop= (df_perc[df_perc['Nans Percentage']>threshold].index.values)
        df_clean= df.drop(cols_to_drop, axis=1) 
        return df_clean


def num_processor(df, chars1= ',\'', chars2= '@\'€%"$'):

    '''num_processor permite introducir un Dataframe y procesa los valores para devolver todos los números en formato float y sin valores
        erróneos como comas o símbolos de dinero. 

        Args:
            df: un Dataframe
            chars1: caracteres que pueden encontrarse como separadores decimales. Default: (, ')
            chars2: caracteres especiales que pueeden aparecer en un dataframe. Default: (@\'€%"$)

        Returns:
            Un nuevo Dataframe con los valores procesados. 

    '''
    nums= ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '.']
    df_return= pd.DataFrame()
    
    for columns in df.columns:
        if df[columns].dtypes== object:
            
            columns_list= df[columns].tolist()

            final_list= []
            for values in columns_list:
                values = str(values)

                for c in chars1:
                    values= values.replace(c, '.')
                for c in chars2:
                    values= values.replace(c,'')
                
                characters= [*values]
                num_list=[]
                for i in characters:
                    if i in nums:
                        num_list.append(i)
                if len(num_list)== 0:
                    num_list.append('No Num')
                final_num= ''.join(num_list)

                try:
                    final_num=float(final_num)
                except ValueError:
                    True
                
                final_list.append(final_num)
                

            n=0
            for categorical in final_list:
                if categorical=='No Num':
                    n+=1
                if n==len(final_list):
                    final_list= (df[columns]).tolist()
         
        else:
            final_list= (df[columns]).tolist()
    
        df_return[columns]= final_list

    for columns in df_return.columns:
        for errors in df_return[columns]:
            if errors== 'No Num':
                print('Posible incongruencia en:')
                print('Columna: ',columns)
                print('Indice: ', df_return[df_return[columns]=='No Num'].index.values[0])
                print()
   
    return df_return
    


def mueve_imagenes (carpeta_fuente, carpeta_train, carpeta_test, n_max=500, split=0.2):

    '''Mueve_imagenes cambia la dirección de las imágenes desde una carpeta original hasta una carpeta de train y otra de test. 
       
       Args:
            carpeta_fuente(str): path de la carpeta original en la que se encuentran las imagenes.
            carpeta_train(str): path de la carpeta de train.
            carpeta_test(str): path de la carpeta de test.
            n_max(int): número máximo de imágenes con las que se quiere trabajar, depende de la disponibilidad. Default= 500.
            split(float): porcentaje de imágenes que se quieren reservar para test. Default=0.2
            
       Returns: None
    '''

    import shutil
    import os

    #Primera parte de la funcion
    imagenes = os.listdir(carpeta_fuente)

    if not os.path.exists(carpeta_train): # esto es para que no se sobreescriba la carpeta
        os.makedirs(carpeta_train)
        print('Carpeta creada: ', carpeta_train)  

    count = 0
    for i, nombreimg in enumerate(imagenes):
        if i < n_max:
            #Copia de la carpeta fuente a la destino
            shutil.copy(carpeta_fuente + '/' + nombreimg, carpeta_train + '/' + str(count) + '.jpg')
            count += 1

    #Segunda parte
    imagenes= os.listdir(carpeta_train)
    if not os.path.exists(carpeta_test): # esto es para que no se sobreescriba la carpeta
        os.makedirs(carpeta_test)
        print('Carpeta creada: ', carpeta_test)  
    
    count = 0
    count = 0
    for i, nombreimg in enumerate(imagenes):
        if i > (np.round(n_max-n_max*split,0)): #En lugar de un numero fijo para dividir en train/test se deja en un porcentaje que por defecto es 0.2
            #Copia de la carpeta fuente a la destino
            shutil.move(carpeta_train + '/' + nombreimg, carpeta_test + '/' + str(count) + '.jpg')
            count += 1



def read_data(path, im_size, class_names):

    ''' read_data lee las imagenes de la carpeta "train" y "test", crea el target a partir de una lista y las convierte en un arreglo de 
        numpy para las X y otro para los targests. Por ultimo, mezcla los datos y las etiquetas de forma aleatoria.

        ARGS:
            path(str): carpeta en la que se encuentran las imagenes.
            im_size(tuple): tamaño de las imagenes.
            class_names(list): nombres de las categorías.
        
        Returns:
            Dos np.arrays, el primero son las X y el segundo las categorías.
    '''


    class_names_label = {class_name:i for i ,class_name in enumerate(class_names)}

    X = []
    y = []

    for folder in os.listdir(path):
        label = class_names_label[folder]
        folder_path = os.path.join(path,folder)
        # Iterar sobre todo lo que haya en path
        for file in os.listdir(folder_path):
            image_path = os.path.join(folder_path,file)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, im_size)
            X.append(image)
            y.append(label)
        
    X_train= np.array(X)
    y_train = np.array(y)

    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    
    return X_train, y_train


def edad(df, columna):

    '''
    Sustituye los valores de fecha de nacimiento por la edad.

    Args:
        df --> Dataframe
        columna --> Columna a la que aplicar la función. Ejemplo: Dataframe['columna']

    Returns:
        Un nuevo Dataframe con la función aplicada. 

    '''



    hoy = date.today()
    columna = pd.to_datetime(columna)

    edad = []
    for i in columna:
        age = hoy.year - i.year - ((hoy.month, hoy.day) < (i.month, i.day))
        edad.append(age)

    df[columna.name] = edad
    
    df[columna.name] = df[columna.name].astype('Int64')
    df.rename(columns = {columna.name : 'edad'}, inplace = True)

    return df


def igualar_strings(df, columna, string_deseado):

    '''
    Cuando en una columna existen strings iguales pero escritos de diferente manera (con acentuación o sin, en mayúsculas o en minúsculas),
    los sustituye por el string que ingresemos manualmente.

    Args:
        df --> Dataframe
        columna --> Columna a la que aplicar la función. Ejemplo: Dataframe['columna'].
        string_deseado --> String por el que queremos sustituir los valores. Ejemplo: "Japón"

    Returns:
        Un dataframe con la función aplicada

    '''
    

    def comparacion(string_original):
        
        if unidecode(string_original.lower()) == unidecode(string_deseado.lower()):
            return string_deseado
        else:
            return string_original

    string_cambiado = columna.apply(comparacion)

    df[columna.name] = string_cambiado

    return df


def outliers(df, columna):

    '''
    Encuentra valores outliers de una columna y elimina las filas en las que se encuentran dichos valores.

    Args:
        df --> Dataframe
        columna --> Columna a la que aplicar la función. Ejemplo: Dataframe['columna']

    Returns:
        Un nuevo Dataframe sin outliers. 

    '''

    # delimitar los quantiles
    quantile1 = np.percentile(columna, 25, interpolation = 'midpoint')
    quantile3 = np.percentile(columna, 75, interpolation = 'midpoint')

    inter_quartile_range = quantile3 - quantile1 

    print('Old Shape: ', df.shape)

    # Límite superior
    upper = np.where(columna >= (quantile3 + 1.5*inter_quartile_range))
    # Límite inferior
    lower = np.where(columna <= (quantile1 - 1.5*inter_quartile_range))

    # Eliminar los outliers
    df.drop(upper[0], inplace = True)
    df.drop(lower[0], inplace = True)
    
    print('New Shape: ', df.shape)

    return df


def porcentaje(columna):

    '''
    Obtiene el porcentaje de aparición de un valor en una columna concreta.

    Args:
        columna --> Columna a la que aplicar la función. Ejemplo: Dataframe['columna'].

    Returns:
        Un dataframe con la columna objetivo y la columna porcentaje.

    '''

    
    porcentaje = round(columna.value_counts(normalize=True) * 100, 2)
    return pd.DataFrame(porcentaje).reset_index().rename(columns={'index' : columna.name, columna.name : 'porcentaje'})


def trimestre(df, string_columna):
    '''
    Agrupa las fechas de una columna datetime por trimestres, sumando los valores del resto de columnas agrupadas.

    Args:
        df --> Dataframe 
        string_columna --> Columna a la que aplicar la función. Ejemplo: "columna"'.

    Returns:
        Un dataframe con la agrupación aplicada

    '''
    

    df_trimestre = df.groupby(pd.Grouper(key = string_columna, freq = '3M')).aggregate(np.sum)
    return df_trimestre


def deteccion_outliers(data,features):
    '''
    Ésta función realiza un bucle para pasar por todas las columnas que indiquemos
    y calcula los valores que se encuentran fuera de los limites de los cuantiles 1 y 3,
    mete en una lista los indices de los valores y cuenta cuantas veces se repiten entre todas las columnas.
    Si el indice lo detecta como outlier en mas de dos columnas mete el indice 
    en la lista de outlier del dataframe.

    Precisa de tener numpy instalado

    Args: 
        -data(DataFrame): Base de datos que tiene todas las columnas que queremos revisar

        -features(columnas): Nombre de todas las columnas que contiene el dataframe a revisar,
        se puede introducir data.columns.

    Return:
        devuelve una lista con los indices que estan fuera de los cuantiles 1 y 3 en 
        mas de dos columnas del DataFrame.

    Agradecimientos a:
     Enes Besinci. Enlace a kaggle-->https://www.kaggle.com/enesbeinci
    '''
    
    outlier_indices = []
    
    for c in features:
        # 1st quartile
        Q1 = np.percentile(data[c],25)
        # 3rd quartile
        Q3 = np.percentile(data[c],75)
        # IQR
        IQR = Q3 - Q1
        # Outlier limite
        outlier_limite = IQR * 1.5
        # detect outlier and their indeces
        outlier_list_col = data[(data[c] < Q1 - outlier_limite) | (data[c] > Q3 + outlier_limite)].index
        # store indeces
        outlier_indices.extend(outlier_list_col)
    
    outlier_indices = Counter(outlier_indices)
    multiples_outliers = list(i for i, v in outlier_indices.items() if v > 2)
    
    return multiples_outliers



def lista_de_listas(lista):
    '''
    La función lista_de_listas crea una lista individual por cada elemento de la lista argumento, 
    pasando asi a una lisa cuyos elementos son listas con un solo elemento

    esto es especialmente util para poder trabajar con strings de varias palabras
    dentro de una lista.

    ejemplo:
        lista = ['estoy programando en python','hola mundo',1234]
        lista_de_listas(lista)
        return [['estoy programando en python'],['hola mundo'],[1234]]

    Args:
        lista: introducir una lista independientemente de los elementos existentes dentro

    Return:
        lista_vacia: una lista de sublistas, cada sublista es un elemento de la lista original.
        
    '''
    lista_vacia = []
    for i in range((len(lista))):
        lista_vacia.append([]) # introduce una lista vacia en lista_vacia por cada vuelta del bucle

    for n,i in enumerate(lista_vacia):
    
        i.append(lista[n]) # en cada valor de lista vacia(i) realiza un append del elemento correspondiente de lista, indicando su indice mediante n

    return lista_vacia


def ratio_nulos(data, features):
    '''
    Función que calcula el porcentaje de valores nulos 
    para cada columna de un dataframe.

    Args:
        data(DataFrame): introduce un dataframe completo
        features(columnas): las columnas del dataframe, podemos usar dataframe.columns.

    return:
        devuelve un dataframe en el cual los indices son los nombres de las columnas 
        y una unica columna con los ratios de valores nulos asignados cada uno a su 
        columna correspondiente. 
    '''

    diccionario ={}
    for c in features:
        
        ratio = (len(data[c][data[c].isnull()== True])/len(data[c]))*100
        
        diccionario[c] = round(ratio,2)

    return pd.DataFrame(diccionario.values(),columns=['null_ratio'],index=diccionario.keys())