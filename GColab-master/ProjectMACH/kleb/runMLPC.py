"""
Testing for Multi-layer Perceptron module (sklearn.neural_network)
DATA: https://www.kaggle.com/flaredown/flaredown-autoimmune-symptom-tracker/home
"""

# Author: Issam H. Laradji
# License: BSD 3 clause

import sys
#import warnings
import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import numpy as np

from numpy.testing import assert_almost_equal, assert_array_equal
from sklearn.datasets import load_boston
from sklearn.datasets import make_regression, make_multilabel_classification
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils.testing import (assert_raises, assert_greater, assert_equal,
                                   assert_false, ignore_warnings)
from sklearn.utils.testing import assert_raise_message
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score

import _pickle as cPickle

#%matplotlib inline

'''
n_inputs (int): Quantos neuronios na camada de entrada.
n_hidden (int): Quantos neuronios na camada oculta.
n_outputs (int): Quantos neuronios na camada de saida.
taxa (flutuacao): A taxa de aprendizado.
Epochs (int): o numero de iteracoes de treinamento.
debug (int): Quanto mais o nivel, mais mensagens sao exibidas.

Retorna:
Um objeto MPPClassification salva, treinada E graficos
'''

##DATA###############################################
isTrain = False

fileData="teste" #Menor
fileTrained = "teste"#fileData
testPercent=0.1

data = pd.read_csv("./{0}.csv".format(fileData))
data.head()

data_train, data_test = train_test_split(data, test_size=testPercent)

#dataOut = ["trackable_value"]
#dataIn = ["age","sex","country","checkin_date","trackable_id","trackable_type","trackable_name"]
dataOut = ["d"]
dataIn = ["BEnh","h","R","Nb","a"]

# user_id,age,sex,country,checkin_date,trackable_id,trackable_type,trackable_name,trackable_value
xData = data_train[dataIn]
yData = data_train[dataOut]

print(yData)

Xtrain = xData.values
Ytrain = yData.values

# Testing data "h","d","R","P0","Nb","a"
xtest = data_test[dataIn]
ytest = data_test[dataOut]

Xtest = xtest.values
Ytest = ytest.values

X = StandardScaler().fit_transform(Xtrain)
Xt = StandardScaler().fit_transform(Xtest)

y = np.array(Ytrain)
y = y.flatten()
 
yt = np.array(Ytest)
yt = yt.flatten() 

##DEEP########################################
#DEBUG
debugData = True
debugDeep = True
saveDNN = True

#CONTROL
fileDNN = './{0}.pkl'.format(fileTrained)
hidden_layer=(1,)#*(1,) tuple, length = n_layers - 2, default (100,)
Epochs = 25000 #25000
learning_rate = 0.005 #0.005
solver =  "adam"   #{'lbfgs', *'sgd', 'adam'}
activation = "relu" #{'identity', 'logistic', 'tanh', *'relu'}
momentumAll = [0]
qtd_batch = 1#"auto"#int, optional, default 'auto' = batch_size=min(200, n_samples)

print("BD\n treino:",len(y),"teste:",len(yt))
    
##PROCESSAMENTO##############################
def criarMLPC():
    print("####CRIANDO MLPClassifier####")
        
    for momentum in momentumAll:
        mlp = MLPClassifier(solver=solver, 
                           hidden_layer_sizes = hidden_layer,
                           activation=activation,
                           learning_rate_init=learning_rate, 
                           random_state=1,
                           batch_size=qtd_batch,
                           max_iter=Epochs, 
                           momentum=momentum)
        
        #TREINO
        mlp.fit(X, y)
                        
        #TESTE
        pred = mlp.predict(Xt)
        
        score = mlp.score(Xt, yt)
        print('score:',score)
                
        #avalia a saida da mlp
        ####VER DISTRIBUICAO DOS DADOS
        if(debugDeep==True):
            debugMLPC(pred)
            
        #assert_greater(score, 0.70)
        #assert_almost_equal(pred, yt, decimal=2)
        
        #######SAVE
        if(saveDNN==True):
            with open(fileDNN, 'wb') as fid:
                cPickle.dump(mlp, fid) 

#CARREGAR MLPR SALVO EM ARQUIVO        
def carregarMLPC():
    print("####CARREGANDO MLPR - NORMAL####")
    
    with open(fileDNN, 'rb') as fid:
        mlp_loaded = cPickle.load(fid)   
    
    #TESTE
    pred = mlp_loaded.predict(Xt)
    
    score = mlp_loaded.score(Xt, yt)
    mse = mean_squared_error(yt, pred)
        
    print('score:',score)
    print('mse:',mse)
        
    debugMLPC(pred)

#DEBUGAR DADOS E RESULTADOS    
def debugMLPC(pred=[]):
    #DEBUG DATA
    if (debugData==True):
        #Avaliar dados
        data.hist(bins=50, figsize=(20,15))
        plt.savefig('./{0}D.eps'.format(fileTrained), format='eps', dpi=1000)
        plt.show()
    
        #Avalia a representatividade dos dados para as CLASSES
        plt.figure(figsize=[10,4])
        sb.heatmap(data.corr())
        plt.savefig('./{0}R.eps'.format(fileTrained), format='eps', dpi=1000)
        plt.show()
        
    #PLOTAR GRAFICO PREDICAO    
    if(debugDeep==True):
        plt.plot(yt)
        plt.plot(pred)
        plt.legend("RP")
        plt.title("Right vs Predicted values")
        plt.savefig('./{0}-{1}-{2}.eps'.format(len(X),hidden_layer, Epochs), format='eps', dpi=1000)    
        plt.show() 
        
if __name__ == '__main__':     
    if(isTrain):
        criarMLPC()
    else:
        carregarMLPC()
        #iotMLPR(Xt, yt)