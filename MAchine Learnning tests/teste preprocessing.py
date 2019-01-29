#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 22:47:51 2017

@author: FlavioTT
"""

#Importar bibliotecas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import dataset
dataset_learnning = pd.read_csv('Data.csv')
x = dataset_learnning.iloc[:,:-1].values
y = dataset_learnning.iloc[:,-1].values

#
