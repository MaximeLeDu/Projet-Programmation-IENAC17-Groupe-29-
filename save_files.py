import numpy as np
def Matrice_coordonnee(shape) :
    M=[]

    n0=shape[0]
    
    for i in range (n0) :
        M.append([i])
    
    for nk in shape[1:] :
        L=len(M)
        for li in range(L) :
            M[li].append(0)
            for k in range(1,nk) :
                elt=M[li].copy()
                elt[-1]=k
                M.append(elt)
    return M

def multi_to_uni (Qsa) :
    shape = list(np.shape(Qsa))
    M = Matrice_coordonnee(shape)
    Qsa_uni=[]
    for coord in M :
        Qsa_uni.append(Qsa[tuple(coord)])
    Qsa_uni.append(shape)
    return Qsa_uni
    
def multi_to_uni_no_zeros (Qsa) :
    shape = list(np.shape(Qsa))
    Qsa_uni_no_zeros=[]
    coord = [0 for i in range (len(shape))]
    tot = 1
    for i in shape :
        tot*=i
    print(tot)
    pourcent=0
    for i in range (tot-1) :
        if int(i/tot*100) ==pourcent+5 :
        pourcent = int(i/tot*100)
            print(pourcent,' sauvegarde\n')
        value = Qsa[tuple(coord)]
        if value != 0.0 :
            Qsa_uni_no_zeros.append([coord.copy(),value.copy()])
        coord_plus_1(coord,shape,0)
    Qsa_uni_no_zeros.append(shape)
    return Qsa_uni_no_zeros

    
def coord_plus_1(coord,shape,i) :
    if coord[i] == shape[i]-1 :
        coord[i]=0
        coord_plus_1(coord,shape,i+1)
    else :
        coord[i]+=1

def  uni_to_multi_no_zeros (Qsa_uni_no_zeros) :
    shape = Qsa_uni_no_zeros.pop()
    Qsa=np.zeros(shape)
    for coord,value in Qsa_uni_no_zeros :
        Qsa[tuple(coord)]=value
    return Qsa
    

def uni_to_multi (Qsa_uni) :
    shape=Qsa_uni.pop()
    M = Matrice_coordonnee(shape)
    Qsa=np.zeros(shape)
    for i in range(len(M)) :
        coord = M[i]
        Qsa[tuple(coord)]=Qsa_uni[i]
    return Qsa

def uni_to_fichier (Qsa_uni,name) :
    Fichier = open(name,'w')
    Fichier.write(str(Qsa_uni))
    Fichier.close()

def fichier_to_uni(name):
    Fichier = open(name,'r')
    Qsa_uni = eval(Fichier.read())
    Fichier.close()
    
    return (Qsa_uni)
    
    
def save_state(name,agent) :
    Fichier = open(name,'w')
    for i in range(len(agent.id_states)):
        key = agent.id_states[i]
        Fichier.write(key + ' ' + str(agent.limites[key][0]) + ' ' + str(agent.limites[key][1]) + ' ' + str(i) + '\n')
    for i in range(len(agent.actions)):
        Fichier.write(str(agent.actions[i]) +'\n')
    Fichier.close()
