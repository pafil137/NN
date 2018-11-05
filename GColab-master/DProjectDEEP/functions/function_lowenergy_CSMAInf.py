'''
Um  Estudo  sobre  o  Consumo  de  Energia  em  RedesAd  HocLineares  Aloha  com  Saltos  Equidistantes
Funcao: Otimizar energia
ENTRADA:potencia  (pt),  tamanho  do  pacote(_Nb),  taxa (_R),  distancia (_dist),  numero  de  saltos (_h), Numero de Nos(h+1).
SAIDA: energia  gasta
'''

#biblioteca para Matematica Python
import math as math
from controle import Controle
import matplotlib.pyplot as plt
import csv

#GLOBAL
ctrol = Controle("csma")

#controle da funcao
_a = ctrol.a
_dist = ctrol.dist
_R = ctrol.R
_Nb = ctrol.Nb
_h = ctrol.h
_n = ctrol.n
_P0i = ctrol.P0i
_P0o = ctrol.P0o

PrxElec = ctrol.PrxElec
Pstart = ctrol.Pstart
Tstart = ctrol.Tstart
PtxElec = ctrol.PtxElec
aamp = ctrol.aamp
bamp = ctrol.bamp
N0 = ctrol.N0
fc = ctrol.fc
c = ctrol.c
Gtant = ctrol.Gtant
Grant = ctrol.Grant
pi = ctrol.pi 
am = ctrol.am
bm = ctrol.bm
L = ctrol.L

####FORMULAS#####
#comprimento de onda (lambda)
lamb = (c/fc)

###consumo  medio  de  energia  por  bit  para  umatransmissao  por  multiplos  saltos
##Ec1 (Formula 6.a - pg2)
C3 = (2*Tstart*Pstart) 

##Ec (Formula 6.b - pg2)
C4 = PtxElec+PrxElec+aamp

##c2 (Formula 10 - pg3)
C2 = (Gtant*Grant*math.pow(lamb,2))/(math.pow(4*math.pi,2)*N0*L)

'''
#############PROCESSAMENTO#############
*numero  de  saltos (_h) = [1,2,4,8]
Energia dissipada-dBmJ/bit(Enh_DBM)
distancia total (_dist) = [0,80]
distancia hop (_distH) = [0,dist]

taxa (_R) = [math.pow(10,3),math.pow(10,4),math.pow(10,5),math.pow(10,6),math.pow(10,7)]
potencia  (P0)
tamanho  do  pacote(_Nb) = [10, 100, 1000, 10000, 100000]
Coeficiente  de  atenuacao(_a) = [2,4]

Numero de Nos(n) = h+1
'''

def functionPower():
    
    #Guardar melhor Configuracao
    _BConfigMatrix = list(range(len(_dist)*len(_h)*len(_a))) 
    _BConfigNb = list(range(len(_dist)*len(_h)*len(_a))) 
    
    i=0
    j=0
    
    for a in _a:        
        for dist in _dist:
            print("####A={0}-Dist={1}####".format(a, dist))
                        
            #Numero de Hops
            for h in _h:        
                
                #energia  gasta
                _BEnh = math.pow(10,7)
                _BConfigR = list(range(len(_R)))
                #Contador R
                j=0
                
                #Taxa de transmissao
                for R in _R:
                    
                    #criar isR: true:
                    if(len(_R)>1):
                        _BEnh = math.pow(10,7)
                        
                    #Tamanho do Pacote
                    for Nb in _Nb:
                                            
                        #Distancia/numero de saltos
                        n = h + 1
                        d = dist/h
                                
                        tau = tauFind(n)
                        Ptr     = 1 - math.pow((1 - tau),n); #Bianchi(10)
                        ps      = (n*tau*math.pow((1-tau),(n - 1)))/Ptr; #Bianchi(11) - probabilidade de sucesso csma/ca
                                                
                        c1 = am*math.pow(d,a)*R*(1+Nb)
                        c2 = 4*bm*C2
                        c3 = math.pow(bamp,2)*math.pow(am,2)*math.pow(d,(2*a))*math.pow((1+Nb),2)
                        c4 = 8*bm*C2*math.pow(R,-1)*((C3*R)+(C4*Nb))*bamp*am*math.pow(d,a)
                        c5 = 4*bm*C2*bamp
                        
                        #P0 = (c1/c2) + R*(math.sqrt(c3+c4)/c5)
                        P0 = _P0i
                        while(P0<_P0o):   
                                                                                        
                            c6 = 1-(am*R*(math.pow(d,a))/float(2*bm*C2*P0))    
                            
                            numeradorEnh = C3*R + (C4+bamp*P0)*Nb
                            denominadorEnh = R*Nb*math.pow(c6,Nb)
                            
                            ##########SAIDAS#############
                            
                            #inconcistencia matematica quando R=2500/(Enh=negativo)
                            #Enh= (numeradorEnh/denominadorEnh)*(1/float(ps))*h #%J/bit
                            prs = ps*math.pow(c6,Nb)
                            EC = 0
                            for k in range(7):
                                EC += k*math.pow(1- prs,k-1)
                            
                            #Energia dissipada em transmissao N hop    
                            E1hop = prs*numeradorEnh*EC/(R*Nb) 
                            Enh= E1hop*h #%J/bit
                            Enh_dbm = 10*math.log10(Enh/0.001) #%dBmJ/bit
                            
                            ####SALVAR DADOS####
                            if(_BEnh > Enh_dbm):                    
                                _BEnh = Enh_dbm 
                                _BConfigR[j] = [_BEnh,h,d,R,P0,Nb,a]
                                #print(_BConfigR[j])
                                
                            P0+=_P0i
                    j+=1
                    
                _BConfigMatrix[i] = _BConfigR
                i+=1
    
    return _BConfigMatrix

def tauFind(n):
    Ret = 7
    
    p=0.1
    thistau = 0.1
    
    p = 1 - math.pow(1-thistau,(n-1))
    thistau = 1/(1 + ((1-p)/(2*(1-math.pow(p,(Ret+1)))))*(findSomatorio(Ret,p)))
    
    return thistau

#%Funcao do tamanho da janela de backoff (2^5) - throughput e praticamente constante; bianchi1 2000
def findSomatorio(Ret, p):
    somatorio = 0
    W = 32  
    for j in range(Ret):
        somatorio = somatorio + (math.pow(p,j))*(math.pow(2,j)*W-1) - (1-math.pow(p,(Ret+1)))
    
    return somatorio

#Gerar treino entrada e saida DNN
def gerarConjuntoTreino(BConfigMatrix):

    file = csv.writer(open("./YXcsma.csv", "wb"))
    
    file.writerow("BEnh,h,d,R,P0,Nb,a")  
    for i in range(len(BConfigMatrix)):
        for j in range(len(_R)):
            file.writerow(BConfigMatrix[i][j])
    
    
def montarGraphR(BConfigMatrix):
    
    legenda = list(range(len(BConfigMatrix)))
    Ehop = list(range(len(BConfigMatrix)))
    
    #Nb = list(range(len(BConfigMatrix)))
            
    #Formar Eihop
    i=0
    for i in range(len(BConfigMatrix)):
        legenda[i] = '{0} Salto(s)'.format(BConfigMatrix[i][1][1])
        
        #GERAR GRAFICOS PARA R
        if(len(_R)>1):
            auxE = list(range(len(_R)))
                        
            for j in range(len(_R)):
                auxE[j] = BConfigMatrix[i][j][0]
                            
            Ehop[i] = auxE
             
    print('\n'.join(map(str, BConfigMatrix)))
    print(legenda)
    plotGraph(legenda, Ehop, _Nb, _R, _a)

#Funcao Plotar Graficos    
def plotGraph(legenda, Ehop, Nb, R, a):
    
    lines = range(len(legenda))
        
    #EM FUNCAO DE R
    x = R
    for i in lines:
        plt.setp(plt.plot(x, Ehop[i]), linewidth=(i+1))
    
    #Gerar figura
    plt.legend(legenda,loc='upper right')
    plt.title('Energia  media  por  bit  em  funcao  da taxa  de  transmissao (R)\n com  tamanho  do  pacote  Nb={0} e a={1}'.format(Nb[0],a[0]))
    plt.ylabel('Eihop - dBmJ/bit')
    plt.xlabel('R - bps')
    
    plt.savefig('./h{0}-d{1}-R{2}-a{3}.eps'.format(len(_h),len(_dist),len(_R),len(_a)), format='eps', dpi=1000)
    plt.show()   
        
#******************************PHYTHON FUNCTIONS*******************************
#************FUNCAO MAIN
if __name__ == '__main__':
    print("#########PARAMETROS##########\n *numero  de  saltos(h)\n Energia dissipada-dBmJ/bit(Enh_DBM)\n distancia hop (_distH)\n distancia(dist)\n taxa(R)\n potencia(P)\n tamanho  do  pacote(Nb)\n Coeficiente  de  atenuacao (a)\n#############################\n")
    print("[Enh_dbm, h, d, R, P, Nb]")
    
    Blist = functionPower()
    
    print("gerarConjuntoTreino")
    gerarConjuntoTreino(Blist)
    
    print("montarGraphR")
    montarGraphR(Blist)
    
    print("####END####")
