def citanjePodataka(datoteka):
    f = open(datoteka, "r")
    E = [[]]
    while(1):
        line = f.readline()
        if(line==""): break
        X=line.split(" ")
        X[0]=int(X[0])
        X[1]=int(X[1])
        while(len(E)<X[0]+1):
            E.append([])
        E[X[0]].append((X[1]))
    return E

datoteka=input()
E=citanjePodataka(datoteka)
print(E)
