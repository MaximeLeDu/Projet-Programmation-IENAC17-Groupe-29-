import numpy as np
import sys
import pickle
import os
import agent as ag

from random import random
from ple import PLE

def choose_game(path) :
    game_list = []
    for game in os.listdir(path) :
        if (game [:2] != "__" and (game [-2:] == "py")) or (os.path.isdir(path+"/"+game) and 'assets' in os.listdir(path+'/'+ game))  :
            game_list.append(game[:-3] if game [-2:] == "py" else game)
    return game_list

def choose_game_state(game) :
    STATES_CHOSEN = []
    for key in list(game.getGameState().keys()) :
        if input ("Prendre en compte " + key + " ? (y/n) \n") == 'y' :
            STATES_CHOSEN.append(key)
    return STATES_CHOSEN

def choose_new_states(game,state_list):
    STATES_CHOSEN = []
    
    for key in list(game.getGameState().keys()) :
        
        if key not in state_list :
            if input ("Prendre en compte " + key + " ? (y/n) \n") == 'y' :
                STATES_CHOSEN.append(key)
    return STATES_CHOSEN

## Initialisation

# Choix du jeu
game_list = choose_game("ple/games")
print (game_list)
game_name = input("Choisissez un jeu parmis la liste. \n")
while game_name not in game_list :
    game_name = input(game_name + " ne fait pas partie de la liste des jeux disponible, entrez un jeu de la liste. \n" )

# Importation du jeu
exec("from ple.games import " + game_name)
print (game_name + " importé")



valide_name = False
while not valide_name :   
    # création du jeu et du PLE
    try :
        game_creation = game_name.lower() + "." + game_name[0].upper() + game_name[1:] + "()"
        game = eval (game_creation)
        valide_name = True
    except :
        game_name = input(game_name+" n'est pas le nom de la class associée au jeu. Il peut s'agir d'une majuscule manquante au milieu du nom. Entrez manuellement le nom de la class associée au jeu.\n")

# Nom des dossier et des fichiers liés au jeu
DOSSIER = game_name + "_texts"
FICHIER_REWARD = DOSSIER + "/" + game_name + "_rewards.p"
FICHIER_STATE = DOSSIER + "/" + game_name + "_states.txt"

# création du PLE
p = PLE(game)

# Choix des états utiles et de la validité de la matrice
redefinition = True
                    
try :
    os.mkdir(DOSSIER) 
    STATES_CHOSEN = choose_game_state(game)
    NEW_STATES_CHOSEN = []
except FileExistsError :
    print ("Le dossier de ce jeu existe déjà")
    with open(FICHIER_STATE) as fichier:
        temp_state = []
        for line in fichier:
            words=line.strip().split()
            if len(words) == 3 :
                temp_state.append(words[0])
    print("Le jeu a été entraîné avec les états suivants : ", temp_state)
    if input("Continuer avec ces états ? (y/n)") == "y" :
        STATES_CHOSEN = temp_state
        NEW_STATES_CHOSEN = choose_new_states(game,temp_state)
        if NEW_STATES_CHOSEN == []:
            redefinition = False
    else :
        STATES_CHOSEN = choose_game_state(game)
        NEW_STATES_CHOSEN = []    
        #On ne garde pas les fichiers sauvegardés quand on change d'états
        os.remove(FICHIER_STATE)
        os.remove(FICHIER_REWARD)
print (DOSSIER,FICHIER_REWARD,FICHIER_STATE)
print ("Les états choisis sont : ",STATES_CHOSEN + NEW_STATES_CHOSEN)


print (p.getActionSet())                    
                    
# création de l'agent
print("creating agent...")
agent = ag.Agent(p.getActionSet(),game.getGameState(),FICHIER_REWARD,FICHIER_STATE,STATES_CHOSEN,NEW_STATES_CHOSEN)
print("agent created")

# Boucle pour définir les limites des états
if redefinition :
    print("Recherche des valeurs limites des états...")
    
    # Condition d'ajout de nouveaux états à un entraînement déjà existant         
    if NEW_STATES_CHOSEN != []:
        print("Ajout de nouveaux états à l'IA...")
        agent.new_states_add(NEW_STATES_CHOSEN)
        print("Ajout terminé")
    for f in range(10000):
        # if the game is over
        if p.game_over():
            p.reset_game()
        
        states = game.getGameState()    
        agent.limite_defineur(states)
        action = agent.random()
        reward = p.act(action)

    print ("Fin de la phase de recherche des limites : ")
    


print (agent.limites)
print (agent.actions)
        

## Boucle d'entrainement

continuer = True
while continuer :
    nb_frames = int(input ("Phase d'entraînement. \nTaille de l'apprentissage ?"))
    pourcent = 0
    reward = 0
    p.force_fps = True
    p.display_screen = False
    p.reset_game()
    print ("Début de la phase d'apprentissage.")
    
    for f in range(nb_frames):
        
        # if the game is over
        if p.game_over():
            p.reset_game()
            game.reset()
        
        states = game.getGameState()
        action = agent.training(reward, agent.discretize(states))
        reward = p.act(action)
        
        # suivi de la progression (tous les 5 pourcents)
        if int(f/nb_frames*100)==5+pourcent :
            pourcent = int(f/nb_frames*100)
            print("Apprentissage fini à  : ",pourcent,'%\n')
    print ("Fin de la phase d'apprentissage.")
    
    
    test = input ("Tester l'IA ? (y/n)")
    if test == "y" :
        f = 0
        p.force_fps = False
        p.display_screen = True
        p.reset_game()
        game.reset()
        while (not p.game_over() or f < 500) and f < 1000  :
            if p.game_over():
                p.reset_game()
                game.reset()
            states = game.getGameState()
            action = agent.AI(agent.discretize(states))
            reward = p.act(action)
            f += 1
    
    continuer = True if input("Continuer l'apprentissage ? (y/n)") == "y" else False
    
## Sauvegarde et fin

print("Sauvegarde de la matrice des récompense et des états...")

#sauvegarde de la matrice des récompense
pickle.dump(agent.rewards,open(FICHIER_REWARD,"wb"))

#sauvegarde des états
with open(FICHIER_STATE,'w') as Fichier:
    for i in range(len(agent.id_states)):
        state = agent.id_states[i]
        Fichier.write(state + ' ' + str(agent.limites[state][0]) + ' ' + str(agent.limites[state][1]) + '\n')
    for i in range(len(agent.actions)):
        Fichier.write(str(agent.actions[i]) +'\n')
        
        
print ("Done")
