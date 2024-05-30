import streamlit as st
from firebase_admin import firestore
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import polars as pl
import csv
import pandas as pd
import numpy as np
import math
import re
import random
#from ydata_profiling import ProfileReport
#import seaborn as sns

#Entrenamiento del modelo
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay, roc_curve, roc_auc_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

#Visualización de datos
import matplotlib.pyplot as plt
import seaborn as sns

import tarfile
import pandas as pd

import seaborn as sns
import csv
import json
#from pickle import dump
#from pickle import load
import requests
from io import BytesIO
import datetime

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import train_test_split
import pickle
from pyspark.sql import SparkSession
import os
# Set the JAVA_HOME variable to the path of your Java installation, and define the exact path where you have installed java. 

os.environ['JAVA_HOME'] = 'C://Program Files//Java//jdk-22'

# Crea una instancia de SparkSession con la configuración especificada
spark = SparkSession.builder\
    .master('local[*]') \
    .config("spark.driver.memory", "10g")\
    .appName("HYB").getOrCreate()
#%matplotlib inline


from surprise import SVD, Dataset, Reader, dump,Dataset, Reader, KNNBasic



import pickle
from sklearn.datasets import load_iris
import joblib
from pyspark.sql.functions import col,when

# open a file, where you stored the pickled data
#file = open('C://Users//Acer//repos//sysrecsongs//dataset//model_CF.pickle', 'rb')

# dump information to that file
#loaded_model = pickle.load(file)

# close the file
#file.close()



def app():
    
    userId=1
    if st.session_state.signout:
            st.title('Recommended for you:' + st.session_state.username)
            
            user=st.session_state.username.split("@",1)[0]
            userId=user.split("_",1)[1]
            
    #db=firestore.client()
    # Define la URL base donde están almacenados los archivos de datos
    url = 'C://Users//Acer//repos//sysrecsongs//dataset//'

    # Carga los datos de entrenamiento y prueba en formato Dataset
  
    # Cargar el archivo CSV
    df_datos = pd.read_csv(f'{url}datos2.csv')
    df_movies= pd.read_csv(f'{url}peliculas_1900_updated_grafos.csv')
    
    #Eliminamos las columnas que no necesitamos para la matriz de recomendación (solo dejamos usuario, pelicula y rating)
    df_datos.drop(columns=["title", "year", "tmdbId", "timestamp"], inplace=True)

    #st.title('AUTOMATIC RECOMENDATIONS: ')
    #st.dataframe(df_datos, use_container_width=True)
    # Dividir los datos en entrenamiento (70%) y prueba (30%)
   
    df_datos_train, df_datos_test = train_test_split(df_datos, test_size=0.3, random_state=42)
    
    
    # Configuración del lector para interpretar las calificaciones en el rango de 1 a 5
    reader = Reader(rating_scale=(1, 5))

    # Carga los conjuntos de datos de entrenamiento y prueba en el formato requerido por la biblioteca Surprise
    train_data = Dataset.load_from_df(df_datos_train, reader)
    test_data = Dataset.load_from_df(df_datos_test, reader)

    df_datos['rating'] = df_datos['rating']*4
    train, test = train_test_split(df_datos, test_size=0.2)

    
    # Obtener las 10 películas más similares a las que el usuario con uid=284574 ha visto
    def obtener_peliculas_similares_por_usuario(uid, n=10):
        # Filtrar las predicciones para el usuario dado
        usuario_filtrado = train[train['userId'] == uid]

        # Ordenar las predicciones por similitud estimada (est)
        usuario_filtrado = usuario_filtrado.sort_values(by='rating', ascending=False)

        # Obtener las n películas más similares
        peliculas_similares = usuario_filtrado.head(n)

        return peliculas_similares

    # Ejemplo de uso: obtener las 10 películas similares para el usuario con uid=284574
    
    peliculas_similares_usuario = obtener_peliculas_similares_por_usuario(uid=userId)
    if(peliculas_similares_usuario.empty):
        peliculas_similares_usuario = obtener_peliculas_similares_por_usuario(uid=284574)
    movie_merge = pd.merge(peliculas_similares_usuario, df_movies, on='movieId')
    #movie_merge=movie_merge.drop(columns=["movieId","userId","year","runtime","vote_average"], inplace=True)
    movie_final=movie_merge[['title','overview','genres','director','actors','release_date','rating','tmdbId']]
   
    
    for i in movie_final['tmdbId']:
        st.components.v1.html(f'<iframe src={i}  width="100%" height="600" style="position:absolute; top:200px; left:0; overflow:hidden; margin-top:-300px;"> </iframe>')
    
    
    st.title('Detail:' )
    st.dataframe(movie_final, use_container_width=True)
    
    #st.page_link("https://www.themoviedb.org/movie/2565-joe-versus-the-volcano", label="themoviedb", icon="🌎") 
   
