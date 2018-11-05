'''
Created on 5 de out de 2018

@author: paulo
'''

import math as math
#range(0.001, 0.16(0.5), 0.001)

############################CONTROLE RNN############################
K = 6                     # numero de dimensoes [h,d,R,P0,Nb,a]
num_H = 25000              # numero de treinamentos
num_test = 5000            # numero de testes
training_epochs = 100      # numero de epocas
trainseed = 0              # definir semente aleatoria para conjunto de treinamento
testseed = 7               # Definir semente aleatoria para o conjunto de teste

class Controle:

    ######VARIAVES DE CONTROLE
    def __init__(self,tipo):
        if(tipo=="aloha"):
            ############################CONTROLE FUNCAO############################
            #########ENTRADAS##############
            ####VARIAVEIS####
            ##Coeficiente  de  atenuacao
            _a = [2]
            #_a = [2, 4]
            #Distancia  entre  fonte  e  destino
            #_dist = [80]
            #_dist = [30,80,150,250]
            _dist = range(10, 250, 5)
            #Taxa de Transmissao (1bps - 1Mbps): 
            #*quanto mais melhor
            #_R = [math.pow(10,3), math.pow(10,4), math.pow(10,5), math.pow(10,6)]
            _R = range(50, int(math.pow(10,6)), 50)
            #Tamanho  do  pacote (1bytes-1mb)
            #_Nb = [1000]
            #_Nb = [1, 10, math.pow(10,2), math.pow(10,3), math.pow(10,4), math.pow(10,5), math.pow(10,6)]
            _Nb = range(1, 7981)#1kbp-8mb
            #Numero  de  saltos  ate  o  destino
            _h = [1, 2, 4, 8]
            #_h = range(1, _dist[0]+1)
            #Numero de Nodos
            n=0
            #_n = h + 1
            _P0i = 0.00001
            _P0o = 0.1
            
            ####CONSTANTES####
            #Potencia  do  circuito  de  recepcao
            PrxElec = 279*math.pow(10,-3)
            #Potencia  para  inicializacao
            Pstart = 58.7*math.pow(10,-3)
            #Tempo  de  inicializacao
            Tstart = 446*math.pow(10,-6)
            #Potencia  do  circuito  de  transmissao
            PtxElec = 151*math.pow(10,-3)
            #Nivel  de  potencia  Eq.  (4)
            aamp = 174*math.pow(10,-3)
            #Constante  de  proporcionalidade  Eq.  (4)
            bamp = 5
            #Densidade  espetral  de  ruido (Watt)
            N0 = math.pow(10, -15.4)*(math.pow(10,-3))
            #Frequencia  da  portadora
            fc = 2.4*math.pow(10,9)
            #Velocidade  da  luz
            c = 3*math.pow(10,8)
            #Ganho  da  antena  de  transmissao
            Gtant = 1
            #Ganho  da  antena  de  recepcao
            Grant = 1
            #PI
            pi = math.pi 
            #Constante de Modulacao(BPSK)
            am = 1
            bm = 2
            #variaveis nao identificadas
            L = 1
            
        if(tipo=="csma"):
            ############################CONTROLE FUNCAO############################
            #########ENTRADAS##############
            ####VARIAVEIS####
            ##Coeficiente  de  atenuacao
            _a = [2]
            #_a = [2, 4]
            #Distancia  entre  fonte  e  destino
            _dist = [80]
            #_dist = [30,80,150,250]
            #_dist = range(10, 250, 5)
            #Taxa de Transmissao (1bps - 1Mbps): 
            #*quanto mais melhor
            #_R = [math.pow(10,3), math.pow(10,4), math.pow(10,5), math.pow(10,6)]
            #_R = range(50, int(math.pow(10,6)), 10000)
            #_R = range(int(math.pow(10,3)),int(math.pow(10,5)),1000)
            _R = (range(int(math.pow(10,3)),int(math.pow(10,4)),int(math.pow(10,3)))
            +range(int(math.pow(10,4)),int(math.pow(10,5)),int(math.pow(10,4)))
            +range(int(math.pow(10,5)),int(math.pow(10,6)-math.pow(10,5)),int(math.pow(10,5)))
            #+range(int(math.pow(10,6)),int(math.pow(10,7)),int(math.pow(10,6)))
            ) 
            #Tamanho  do  pacote (1bytes-1mb)
            #_Nb = [1000]
            #_Nb = [1, 10, math.pow(10,2), math.pow(10,3), math.pow(10,4), math.pow(10,5), math.pow(10,6)]
            #_Nb = range(1, 100)
            _Nb = range(1, 7981)#1kbp-8mb
            #Numero  de  saltos  ate  o  destino
            _h = [1, 2, 4, 8]
            #_h = range(1, _dist[0]+1)
            #Numero de Nodos
            n=0
            #_n = h + 1
            _P0i = 0.00001
            _P0o = 0.1
            
            ####CONSTANTES####
            #Potencia  do  circuito  de  recepcao
            PrxElec = 279*math.pow(10,-3)
            #Potencia  para  inicializacao
            Pstart = 58.7*math.pow(10,-3)
            #Tempo  de  inicializacao
            Tstart = 446*math.pow(10,-6)
            #Potencia  do  circuito  de  transmissao
            PtxElec = 151*math.pow(10,-3)
            #Nivel  de  potencia  Eq.  (4)
            aamp = 174*math.pow(10,-3)
            #Constante  de  proporcionalidade  Eq.  (4)
            bamp = 5
            #Densidade  espetral  de  ruido (Watt)
            N0 = math.pow(10, -15.4)*(math.pow(10,-3))
            #Frequencia  da  portadora
            fc = 2.4*math.pow(10,9)
            #Velocidade  da  luz
            c = 3*math.pow(10,8)
            #Ganho  da  antena  de  transmissao
            Gtant = 1
            #Ganho  da  antena  de  recepcao
            Grant = 1
            #PI
            pi = math.pi 
            #Constante de Modulacao(BPSK)
            am = 1
            bm = 2
            #variaveis nao identificadas
            L = 1
            
        #controle da funcao
        self.a = _a
        self.dist = _dist
        self.R = _R
        self.Nb = _Nb
        self.h = _h
        self.P0i = _P0i
        self.P0o = _P0o
        self.n = n
        self.PrxElec = PrxElec
        self.Tstart = Tstart
        self.Pstart = Pstart
        self.PtxElec = PtxElec
        self.aamp = aamp
        self.bamp = bamp
        self.N0 = N0
        self.fc = fc
        self.c = c
        self.Gtant = Gtant
        self.Grant = Grant
        self.pi = pi 
        self.am = am
        self.bm = bm
        self.L = L
        
        #VARIAVES DNN
        self.K = K                     
        self.num_H = num_H
        self.num_test = num_test
        self.training_epochs = training_epochs
        self.trainseed = trainseed              
        self.testseed = testseed               
        