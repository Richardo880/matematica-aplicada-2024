#TRABAJO REALIZADO POR RICARDO TOLEDO Y ZINRI BOBADILLA

import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import skfuzzy as fuzz
import numpy as np
import time
import re

# Definir el diccionario de contracciones
contractions_dict_space = {
    r"\bi m\b": "I am",
    r"\bi ve\b": "I have",
    r"\bi ll\b": "I will",
    r"\bi d\b": "I would",
    r"\byou re\b": "you are",
    r"\byou ve\b": "you have",
    r"\byou ll\b": "you will",
    r"\byou d\b": "you would",
    r"\bhe s\b": "he is",
    r"\bhe d\b": "he would",
    r"\bshe s\b": "she is",
    r"\bshe d\b": "she would",
    r"\bwe re\b": "we are",
    r"\bwe ve\b": "we have",
    r"\bwe ll\b": "we will",
    r"\bwe d\b": "we would",
    r"\bthey re\b": "they are",
    r"\bthey ve\b": "they have",
    r"\bthey ll\b": "they will",
    r"\bthey d\b": "they would",
    r"\bit s\b": "it is",
    r"\bit d\b": "it would",
    r"\bit ll\b": "it will",
    r"\bcan t\b": "cannot",
    r"\bwon t\b": "will not",
    r"\bdon t\b": "do not",
    r"\bdoesn t\b": "does not",
    r"\bdidn t\b": "did not",
    r"\bcouldn t\b": "could not",
    r"\bshouldn t\b": "should not",
    r"\bwouldn t\b": "would not",
    r"\bweren t\b": "were not",
    r"\bisn t\b": "is not",
    r"\baren t\b": "are not",
    r"\bhaven t\b": "have not",
    r"\bhasn t\b": "has not",
    r"\bhadn t\b": "had not",
    r"\bn t\b": " not"
}


# Función para expandir contracciones
def expand_contractions(text):
    for contraction, expanded in contractions_dict_space.items():
        text = re.sub(contraction, expanded, text, flags=re.IGNORECASE)
    return text

# Función de limpieza
def limpiar(data):
    # Aplicar la expansión de contracciones a cada tweet
    data['tweet'] = data['tweet'].apply(expand_contractions)

# Cargar el dataset
data = pd.read_csv("test_data.csv")
data.rename(columns={"sentence": "tweet"}, inplace=True)

# Aplicar la función de limpieza
limpiar(data)

# Transformar sentimiento numérico a textual
def trans_sentiment(score):
    if score == 0:
        return "Negativo"
    else:
        return "Positivo"
data['sentiment'] = data['sentiment'].apply(lambda row:trans_sentiment(row))
data.rename(columns={"sentiment": "sentimiento"}, inplace=True)


# Inicializar el analizador de sentimiento usando VADER en este caso
sia = SentimentIntensityAnalyzer()

# Modulo para calcular puntajes de sentimiento positivo y negativo
def sentiment_score(sentence):
    scores = sia.polarity_scores(sentence)
    if (scores['pos']==1):
        scores['pos']=0.9 
    else:
        scores['pos']=round(scores['pos'],1)
   
    if (scores['neg']==1):
        scores['neg']=0.9
    else:
        scores['neg']=round(scores['neg'],1)

    
    return scores['pos'], scores['neg'],scores['neu']

# Calcular puntajes y agregar al dataframe
data[['puntaje_positivo', 'puntaje_negativo','puntaje_neutral']] = data['tweet'].apply(lambda x: pd.Series(sentiment_score(x)))


# Función de membresía triangular
def triangular_membership(x, d, e, f):
    if x <= d:
        return 0
    elif d < x <= e:
        return (x - d) / (e - d)
    elif e < x < f:
        return (f - x) / (f - e)
    else:
        return 0
    

# Calcular min, max y mid
pos_min, pos_max = data['puntaje_positivo'].min(), data['puntaje_positivo'].max()
neg_min, neg_max = data['puntaje_negativo'].min(), data['puntaje_negativo'].max()
pos_mid = (pos_min + pos_max) / 2
neg_mid = (neg_min + neg_max) / 2


# Fuzzificacion
def fuzzy(tweet, min_val, mid_val, max_val,min_valNeg, mid_valNeg, max_valNeg):
    
    tInicio_fuzzy = time.perf_counter()
    low = triangular_membership(tweet['puntaje_positivo'], 0, min_val, mid_val)
    medium = triangular_membership(tweet['puntaje_positivo'], min_val, mid_val, max_val)
    high = triangular_membership(tweet['puntaje_positivo'], mid_val, max_val, 10)

    lowNeg = triangular_membership(tweet['puntaje_negativo'], 0, min_valNeg, mid_valNeg)
    mediumNeg = triangular_membership(tweet['puntaje_negativo'], min_valNeg, mid_valNeg, max_valNeg)
    highNeg = triangular_membership(tweet['puntaje_negativo'], mid_valNeg, max_valNeg, 10)

    tFin_fuzzy = time.perf_counter()
   
    return (tFin_fuzzy-tInicio_fuzzy)


# Aplicar la función fuzzy a cada fila
data['tiempo_fuzzy'] = data.apply(lambda row: fuzzy(row, pos_min, pos_mid, pos_max, neg_min, neg_mid, neg_max), axis=1)


# Generar valores universales
x_p = np.arange(0, 1, 0.1)
x_n = np.arange(0, 1, 0.1)
x_op = np.arange(0, 10, 1)

# Generar fuzzy membership
pos_bajo = fuzz.trimf(x_p, [0, 0, 0.5])
pos_mid = fuzz.trimf(x_p, [0, 0.5, 1])
pos_alto = fuzz.trimf(x_p, [0.5, 1, 1])
neg_bajo = fuzz.trimf(x_n, [0, 0, 0.5])
neg_mid = fuzz.trimf(x_n, [0, 0.5, 1])
neg_alto = fuzz.trimf(x_n, [0.5, 1, 1])
op_Neg = fuzz.trimf(x_op, [0, 0, 5])  # Escala : Neg Neu Pos
op_Neu = fuzz.trimf(x_op, [0, 5, 10])
op_Pos = fuzz.trimf(x_op, [5, 10, 10])

def defuzzy(tweet, op_Neg, op_Neu, op_Pos,x_op):
    ini=time.perf_counter()
    pos_niv_bajo = fuzz.interp_membership(x_p, pos_bajo, tweet['puntaje_positivo'])
    pos_niv_mid= fuzz.interp_membership(x_p, pos_mid, tweet['puntaje_positivo'])
    pos_niv_alto = fuzz.interp_membership(x_p, pos_alto, tweet['puntaje_positivo'])
    
    neg_niv_bajo = fuzz.interp_membership(x_n, neg_bajo, tweet['puntaje_negativo'])
    neg_niv_mid= fuzz.interp_membership(x_n, neg_mid, tweet['puntaje_negativo'])
    neg_niv_alto = fuzz.interp_membership(x_n, neg_alto, tweet['puntaje_negativo'])
    
    # Reglas de activacion
    # el operador AND indica que tomamos el minimo entre los 2
    regla_activ_1 = np.fmin(pos_niv_bajo, neg_niv_bajo)
    regla_activ_2 = np.fmin(pos_niv_mid, neg_niv_bajo)
    regla_activ_3 = np.fmin(pos_niv_alto, neg_niv_bajo)
    regla_activ_4 = np.fmin(pos_niv_bajo, neg_niv_mid)
    regla_activ_5 = np.fmin(pos_niv_mid, neg_niv_mid)
    regla_activ_6 = np.fmin(pos_niv_alto, neg_niv_mid)
    regla_activ_7 = np.fmin(pos_niv_bajo, neg_niv_alto)
    regla_activ_8 = np.fmin(pos_niv_mid, neg_niv_alto)
    regla_activ_9 = np.fmin(pos_niv_alto, neg_niv_alto)


    # Ahora aplicamos esto recortando la parte superior de la salida correspondiente
    # funcion de membresia  con np.fmin
    neg1=np.fmax(regla_activ_4,regla_activ_7)
    neg2=np.fmax(neg1,regla_activ_8)     
    op_activation_bajo = np.fmin(neg2,op_Neg)

    neu1=np.fmax(regla_activ_1,regla_activ_5)
    neu2=np.fmax(neu1,regla_activ_9)     
    op_activation_mid = np.fmin(neu2,op_Neu)

    pos1=np.fmax(regla_activ_2,regla_activ_3)
    pos2=np.fmax(pos1,regla_activ_6)   
    op_activation_alto = np.fmin(pos2,op_Pos)

   
   # Agrega las tres funciones de membresía de salida juntas
    agg = np.fmax(op_activation_bajo,np.fmax(op_activation_mid, op_activation_alto))

    # Calcular el resultado defuzzificado
    op = fuzz.defuzz(x_op, agg, 'centroid')
    valor=round(op,2)

    resul=""
    if 0<(valor)<3.33:    
        resul="Negativo"   
    elif 3.34<(valor)<6.66:
        resul="Neutral" 
    elif 6.67<(valor)<10: 
        resul="Positivo"
    fin=time.perf_counter()
    return valor,resul,fin-ini



data[['defuzzificación', 'sentimiento_defuzz','tiempo_defuzzificar']] = data.apply(lambda row: pd.Series(defuzzy(row, op_Neg, op_Neu, op_Pos,x_op)), axis=1)
data['tiempo_total']=data.apply(lambda row: row['tiempo_fuzzy']+row['tiempo_defuzzificar'],axis=1)
data.to_csv('resultado.csv',index=False)