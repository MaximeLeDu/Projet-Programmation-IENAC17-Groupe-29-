import math
import random
from ple import PLE
import agent as ag
from ple.games import Catcher

def genetique(générations,N,proba,game,STATES_CHOSEN):

    p=PLE(game,force_fps = True,display_screen = False)
    Fonctions = []
    for j in range(N):
        Fonctions.append([calcul_poids(1/(j+1.1),1/(j+1.5),1/(j+1.3),4*(j+1),p,STATES_CHOSEN),1/(j+1.1),1/(j+1.5),1/(j+1.3),4*(j+1)])
#Cette boucle initialise N solutions initiales.
        
    for j in range(générations):
        n=len(Fonctions)
        for l in range(n-1):
            (a,b,c,d,a0,b0,c0,d0)=croisement(Fonctions[l][1],Fonctions[l][2],Fonctions[l][3],Fonctions[l][4],Fonctions[l+1][1],Fonctions[l+1][2],Fonctions[l+1][3],Fonctions[l+1][4])
            (a,b,c,d)=mutation(a,b,c,d,proba)
            (a0,b0,c0,d0)=mutation(a0,b0,c0,d0,proba)
            Fonctions.append([calcul_poids(a,b,c,d,p,STATES_CHOSEN),a,b,c,d])
            Fonctions.append([calcul_poids(a0,b0,c0,d0,p,STATES_CHOSEN),a0,b0,c0,d0])
        (a,b,c,d,a0,b0,c0,d0)=croisement(Fonctions[0][1],Fonctions[0][2],Fonctions[0][3],Fonctions[0][4],Fonctions[n-1][1],Fonctions[n-1][2],Fonctions[n-1][3],Fonctions[n-1][4])
        (a,b,c,d)=mutation(a,b,c,d,proba)
        (a0,b0,c0,d0)=mutation(a0,b0,c0,d0,proba)
        Fonctions.append([calcul_poids(a,b,c,d,p,STATES_CHOSEN),a,b,c,d])
        Fonctions.append([calcul_poids(a0,b0,c0,d0,p,STATES_CHOSEN),a0,b0,c0,d0])
        epuration(Fonctions,N)
#Cette boucle exécute la partie génétique de l'algorithme : le croisement, la mutation, l'évaluation par la fonction de poids et la sélection.    
    return (Fonctions[0][1],Fonctions[0][2],Fonctions[0][3],Fonctions[0][4],Fonctions[0][0])
#Les meilleures données sont renvoyées, avec le poids associé.


def fonction_de_tri(T):
    return T[0]
#Le tableau des coefficients est trié selon la première composante de chaque élément, c'est-à-dire leur poids, ce qui permet de faciliter la sélection des meilleures solutions.

def epuration(T,N):
    T.sort(key=fonction_de_tri,reverse = True)
    print(T)
    i,j=0,len(T)-1
    while i<j:
        if T[i]==T[i+1]:
            del T[i]
            j=j-1
        else:
            i=i+1
    n=len(T)
    if n>N:
        del T[N:n]
#On ne sélectionne que les N meilleures solutions.


def mutation(a,g,e,n,p):
    if random.random()<p:
        if random.random()<0.5:
            a0=a+0.1*random.random()
        else:
            a0=a-0.1*random.random()
        if random.random()<0.5:
            g0=g+0.1*random.random()
        else:
            g0=g-0.1*random.random()
        if random.random()<0.5:
            e0=e+0.1*random.random()
        else:
            e0=e-0.1*random.random()
        if random.random()<0.5:
            n0=int(n+5*random.random())
        else:
            n0=int(n-5*random.random())
        if a0>0 and a0<1 and g0>0 and g0<1 and  e0>0 and e0<1 and n0>0:
            return (a0,g0,e0,n0)
    return(a,g,e,n)


def croisement(alpha0,gamma0,eps0,norm0,alpha1,gamma1,eps1,norm1):
    croi1=gamma0-random.random()*(gamma0-gamma1)
    croi2=gamma1-random.random()*(gamma0-gamma1)
    croi3=int(norm0-random.random()*(norm0-norm1))
    croi4=int(norm1-random.random()*(norm0-norm1))
    if croi3>0 and croi4>0 and croi1>0 and croi1<1 and croi2>0 and croi2<0:
        return(alpha0,croi1,eps1,croi3,alpha1,croi2,eps0,croi4)
    return(alpha0,gamma0,eps1,norm1,alpha1,gamma1,eps0,norm0)
    
def calcul_poids(a,g,e,n,p,STATES_CHOSEN):
    #On fait un entraînement avec les paramètres, le poids est donné par la somme des récompenses sur une partie de taille fixée.
    agent = ag.Agent(p.getActionSet(),p.game.getGameState(),None,None,STATES_CHOSEN,[],a,g,e,n)
    score = 0
    reward = 0
    for f in range(10000):
    
        if p.game_over():
            p.reset_game()
        
        states = p.game.getGameState()    
        agent.limite_defineur(states)
        action = agent.random()
        reward = p.act(action)
    
    for f in range(1000000):
        
        # if the game is over
        if p.game_over():
            p.reset_game()
            p.game.reset()
        
        states = p.game.getGameState()
        action = agent.training(reward, agent.discretize(states))
        reward = p.act(action)
        
    for f in range(50000):
        
        if p.game_over():
            p.reset_game()
            p.game.reset()
        
        states = p.game.getGameState()
        action = agent.AI(agent.discretize(states))
        reward = p.act(action)
        score+=reward
        
    return score
    
print("Resultat :",genetique(7,7,0.6,Catcher(),["fruit_x","player_x"]))
