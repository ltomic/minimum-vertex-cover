from bitarray import bitarray #ovo mi se cinilo fora
from random import randint
from random import random


def stupanj(x, E):  #prebaciti E u globalnu varijablu? isto za W
    return len(E[x])

def visak(x, stanje, E):    #provjeriti redudanciju elementa u rjesenju

    if(stanje[x]==0): return False

    for i in E[x]:
        if stanje[i] == 0:
            return False

    return True

    


def citanjePodataka(datoteka):  #uvesti podatke iz .txt

    f = open(datoteka, "r")

    E = [[]]

    # B = [] BRIDOVI
    # W = [] TEZINE CE BITI POTREBNE (VIDI SLJEDECU FUNKCIJU)

    while(1):

        line = f.readline()

        if(line==""): break

        X=line.split(" ")

        # B.append(X)

        X[0]=int(X[0])
        X[1]=int(X[1])

        while(len(E)<X[0]+1):   #mozda nam je vec poznat broj vrhova
            E.append([])        #iznimno, W.append(1) za netezinski graf

        if (X[1] not in E[X[0]]):   #nekada nepotrebno
            E[X[0]].append((X[1]))

        while(len(E)<X[1]+1):   #mozda nam je vec poznat broj vrhova
            E.append([])        #iznimno, W.append(1) za netezinski graf

        if (X[0] not in E[X[1]]):
            E[X[1]].append((X[0]))

    f.close()

    return E    #B, W, DOBRO BI NAM DOSAO INTEGER BROJ VRHOVA (UMJESTO LEN() KASNIJE)


def redukcija(stanje, E): #izbaciti visak iz rjesenja
                             #OVDJE ZELIMO TEZINE, W KAO ARGUMENT?
    VISKOVI = []    #ovo bi mogao biti bitarray
    mx = 0 #za najveci omjer tezine i stupnja

    brv = len(stanje)   #broj vrhova

    W = [1] * brv #privremena solucija, netezinski graf
    
    for i in range(brv):
        if visak(i, stanje, E):
            VISKOVI.append(i)   #ako je bitarray, append(1) else: append(0)

    while(VISKOVI!=[]):
        for i in VISKOVI:       
            if(W[i]/stupanj(i, E)) > mx:
                mx = W[i]/stupanj(i, E)
                ix = i              #statisticki najgori vrh u rjesenju

        if(random()>0.5):       #o ovom 0.5 treba razmisliti, to je p_sc kod Singha
            VISKOVI.remove(ix)
            stanje[ix] = 0
        else:                       #ovako Singh izbaci random vrh ako ne izbaci najgori
            ix = VISKOVI[randint(0, len(VISKOVI)-1)]
            VISKOVI.remove(ix)
            stanje[ix] = 0

        for i in VISKOVI:           #sada neki elementi mozda vise nisu viskovi
            if (not visak(i, stanje, E)):
                VISKOVI.remove(i)
        mx = 0

    return stanje
            
        
    
def generiranje(E): #generira pocetno rjesenje

    stanje = bitarray()

    for i in range(len(E)):
        if(stupanj(i, E)==0):
            stanje.append(0)
        else:
            stanje.append(randint(0, 1)) #mozda neki bolji random?

    return stanje


datoteka=input()
E=citanjePodataka(datoteka)
S=generiranje(E)
print(S.count(1))
S=redukcija(S, E)
print(S.count(1))
