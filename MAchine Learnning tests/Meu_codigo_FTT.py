#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 18:42:38 2018

@author: FlavioTT
"""

#importing libs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.set_printoptions(threshold = np.nan)

#importing dataset
dataset = pd.read_csv('Data.csv')

#definning features and independent variables
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values
#x1 = dataset.iloc[:, :-1].values

#dealing with missing data and using sklearn
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0) 
#axis define se quer substituir pela media das colunas ou média das linhas
imputer = imputer.fit(x[:, 1:3]) #substituir a coluna 1 [1,3[
x[:, 1:3] = imputer.transform(x[:, 1:3])

#Categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])

#the problem is when you transform the columms country in numbers [0 - 2] 
#the bigger the number, the biggers is his wheigt of that category
#it's better transform them in dummy variables

onehotencoder = OneHotEncoder(categorical_features = [0]) 
x = onehotencoder.fit_transform(x).toarray()
#x1 = onehotencoder.fit_transform(x).toarray() #testes criando dummy diretamente

#Adjusting the target variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#Splitting in trainning and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#Normalizaçao das variáveis explicativas para modelos baseados em distancia euclidiana
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
