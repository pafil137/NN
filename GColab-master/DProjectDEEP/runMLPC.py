"""
Testing for Multi-layer Perceptron module (sklearn.neural_network)
"""

# Author: Issam H. Laradji
# License: BSD 3 clause

#MLP
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

#Math AUX
import math
import numpy as np

#DATA USER
import pandas as pd
import seaborn as sb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#Metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score

#Save MLP
import _pickle as cPickle

#PLOT GRAPH
import matplotlib.pyplot as plt

'''
n_inputs (int): Quantos neuronios na camada de entrada.
n_hidden (int): Quantos neuronios na camada oculta.
n_outputs (int): Quantos neuronios na camada de saida.
taxa (flutuacao): A taxa de aprendizado.
Epochs (int): o numero de iteracoes de treinamento.
debug (int): Quanto mais o nivel, mais mensagens sao exibidas.

Retorna:
Um objeto MPPRegressor pronto para ser treinado E graficos
'''

##DATA###############################################
isTrain = False

fileData="YXsmall" #Menor
fileTrained = "YXsmall"#fileData
testPercent=0.1

data = pd.read_csv("./ConjTreino/{0}.csv".format(fileData))
data.head()

data_train, data_test = train_test_split(data, test_size=testPercent)

dataOut = ["d"]
dataIn = ["BEnh","R","Nb","a"]

# Training data ["h","d","R","P0","Nb","a"]
xData = data_train[dataIn]
yData = data_train[dataOut]

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
debugData = False
debugDeep = True
saveDNN = True

#CONTROL
fileDNN = './trainedDNN/{0}.pkl'.format(fileTrained)
hidden_layer=(1,)#*(1,) tuple, length = n_layers - 2, default (100,)
Epochs = 25000 #25000
learning_rate = 0.005 #0.005
solver =  "adam"   #{'lbfgs', *'sgd', 'adam'}
activation = "relu" #{'identity', 'logistic', 'tanh', *'relu'}
momentumAll = [0]
qtd_batch = "auto"#"auto"#int, optional, default 'auto' = batch_size=min(200, n_samples)

print("BD\n treino:",len(y),"teste:",len(yt))
    
##PROCESSAMENTO##############################
def criarMLPC():
    print("####CRIANDO MLPC####")
    
    for momentum in momentumAll:
        mlp = MLPClassifier(solver=solver, 
                           hidden_layer_sizes = hidden_layer,
                           activation=activation,
                           learning_rate_init=learning_rate, 
                           random_state=1,
                           batch_size=qtd_batch, 
                           momentum=momentum,
                           max_iter=Epochs)
        
        #TREINO
        n_classes = np.unique(y)
        for i in range(Epochs):
            mlp.partial_fit(X, y, n_classes)
            print((100*i/Epochs),"%")
        
        #TESTE
        pred = mlp.predict(Xt)
        
        score = mlp.score(X, y)
        print(score)
        
        #avalia a saida da mlp
        ####VER DISTRIBUICAO DOS DADOS
        if(debugDeep==True):
            debugMLPC(pred)
            
        #######SAVE
        if(saveDNN==True):
            with open(fileDNN, 'wb') as fid:
                cPickle.dump(mlp, fid) 

#CARREGAR MLPR SALVO EM ARQUIVO        
def carregarMLPC():
    print("####CARREGANDO MLPC - NORMAL####")
    
    with open(fileDNN, 'rb') as fid:
        mlp = cPickle.load(fid)   
    
    #TESTE
    pred = mlp.predict(Xt)
    
    score = mlp.score(Xt, yt)
    print(score)
    
    debugMLPC(pred)


##SAIDA############################## 
def debugMLPC(pred=[]):
    #DEBUG DATA
    if (debugData==True):
        #Avaliar dados
        data.hist(bins=50, figsize=(20,15))
        plt.savefig('./outPut/{0}DC.eps'.format(file), format='eps', dpi=1000)
        plt.show()
    
        #Avalia a representatividade dos dados para as CLASSES
        plt.figure(figsize=[10,4])
        sb.heatmap(data.corr())
        plt.savefig('./outPut/{0}RC.eps'.format(file), format='eps', dpi=1000)
        plt.show()
        
    #PLOTAR GRAFICO PREDICAO    
    if(debugDeep==True):
        accuracy = accuracy_score(yt, pred)
        print('accuracy:',accuracy)
        
        print(yt)
        print(pred)
        
        #n_classes = np.unique(yt)
        
        plt.plot(pred, 'ro')
        plt.plot(yt, "b+")
        plt.axis([0, 2, 0, 250])
        
        plt.legend("RP")
        plt.title("Real Values vs Predicted Points")
            
        #plt.savefig('./outPut/{0}-{1}-{2}C.eps'.format(len(X),hidden_layer, Epochs), format='eps', dpi=1000)    
        plt.show()
 
if __name__ == '__main__':     
    if(isTrain):
        criarMLPC()
    else:
        carregarMLPC()