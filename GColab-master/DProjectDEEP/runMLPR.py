"""
Testing for Multi-layer Perceptron module (sklearn.neural_network)
"""

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
from sympy.logic.boolalg import false

##PRECONFIG####################################
#CONFIGURATION
solverC     = ["adam", "sgd",      "adam", "sgd",      "sgd",  "sgd"]
activationC = ["relu", "logistic", "tanh", "identity", "relu", "tanh"]

#DATA
nameData = ["YXsmall","YXlite","YXmid","YXbig","YXcomplete","YXcompleteS", "YXramdom"]

##INPUT########################################    
#TRAINO OU TESTE
isTrain = False
config = 2 #2 BESTCONFIG: adam-tanh
configDataTeste = 5 #5 COMPLETES
configDataTrain = 4 #4 COMPLETE

qtd_batch = 1#BEST: 1 "auto"#int, optional, default 'auto' = batch_size=min(200, n_samples)

solver =  solverC[config]
activation = activationC[config]

fileData=nameData[configDataTeste]
fileTrained = nameData[configDataTrain]+"{0}1".format(config)

fileDNN = './trainedDNN/{0}.pkl'.format(fileTrained)
hidden_layer=(1,)#*(1,) tuple, length = n_layers - 2, default (100,)
Epochs = 25000 #25000
learning_rate = 0.005 #0.005
momentumAll = [0]

#DEBUG
debugData = False
debugDeep = True
saveDNN = True

#CONJ. TREINO/TESTE
if(isTrain):
    testPercent=0.01
else:
    testPercent=0.9

##DATA###############################################
data = pd.read_csv("./ConjTreino/{0}.csv".format(fileData))
data.head()

data_train, data_test = train_test_split(data, test_size=testPercent)

dataOut = ["BEnh"]
dataIn = ["h","d","R","Nb","a"]

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

##PROCESSAMENTO##############################
#DEEP
print("BD\n treino:",len(y),"teste:",len(yt))
def criarMLPR():
    print("####CRIANDO MLPR####")
    
    for momentum in momentumAll:
        
        mlp = MLPRegressor(solver=solver, 
                           hidden_layer_sizes = hidden_layer,
                           activation=activation,
                           learning_rate_init=learning_rate, 
                           random_state=1,
                           batch_size=qtd_batch, 
                           momentum=momentum)
        
        #TREINO
        for i in range(Epochs):
            mlp.partial_fit(X, y)
            print((100*i/Epochs),"%")
                
        #TESTE
        pred = mlp.predict(Xt)
        
        score = mlp.score(Xt, yt)
        print('score:',score)
                
        #assert_greater(score, 0.70)
        #assert_almost_equal(pred, yt, decimal=2)
        
        #######SAVE
        if(saveDNN==True):
            with open(fileDNN, 'wb') as fid:
                cPickle.dump(mlp, fid) 
                
        #avalia a saida da mlp
        ####VER DISTRIBUICAO DOS DADOS
        if(debugDeep==True):
            debugMLPR(pred)

#CARREGAR MLPR SALVO EM ARQUIVO        
def carregarMLPR():
    print("####CARREGANDO MLPR - NORMAL####")
    
    with open(fileDNN, 'rb') as fid:
        mlp_loaded = cPickle.load(fid)   
    
    #TESTE
    pred = mlp_loaded.predict(Xt)
    
    score = mlp_loaded.score(Xt, yt)
    mse = mean_squared_error(yt, pred)
        
    print('score:',score)
    print('mse:',mse)
    
    
    debugMLPR(pred)

#TESTE DE FUNCIONAMENTO NO IOT
def iotMLPR(Xp, yp):
    print("####CARREGANDO MLPR - FOR IOT####")
    
    with open(fileDNN, 'rb') as fid:
        mlp_loaded = cPickle.load(fid)   
    
    #TESTE
    pred = mlp_loaded.predict(Xp)
    
    score = mlp_loaded.score(Xp, yp)
    mse = mean_squared_error(yp, pred)
    #predScore = precision_score(yp, pred, average='micro')
    
    print('score:',score)
    print('mse:',mse)
     
    plt.plot(yt)
    plt.plot(pred, 'ro')
    plt.legend("RP")
    plt.title("Real Values vs Predicted Points CONFIG:{0}".format(config))
    plt.show()



##SAIDA##############################   
def debugMLPR(pred=[]):
    #DEBUG DATA
    if (debugData==True):
        #Avaliar dados
        data.hist(bins=50, figsize=(20,15))
        plt.savefig('./ConjTreino/{0}DR.eps'.format(fileData), format='eps', dpi=1000)
        plt.show()
    
        #Avalia a representatividade dos dados para as CLASSES
        plt.figure(figsize=[10,4])
        sb.heatmap(data.corr())
        plt.savefig('./ConjTreino/{0}.eps'.format(fileData), format='eps', dpi=1000)
        plt.show()
        
                
    #PLOTAR GRAFICO PREDICAO    
    if(debugDeep==True):
        plt.plot(yt)
        plt.plot(pred)
        plt.legend("RP")
        plt.title("Right vs Predicted values")
        #plt.savefig('./outPut/{0}-{1}-{2}R.eps'.format(len(X),hidden_layer, Epochs), format='eps', dpi=1000)    
        plt.show()
    
if __name__ == '__main__':     
    if(isTrain):
        criarMLPR()
    else:
        carregarMLPR()
        #iotMLPR(Xt, yt)
