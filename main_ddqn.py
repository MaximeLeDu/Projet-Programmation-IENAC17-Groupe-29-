import os
from ple import PLE
import random
import numpy as np
import essai
from ple import PLE
from ple.games import Pong


def train(nb_parties,agent,p):
    for e in range(nb_parties):
        score = 0
        p.reset_game()
        state = list(p.game.getGameState().values())
        state = np.reshape(state, [1, agent.state_size])

        while not p.game_over():

            # get action for the current state and go one step in environment
            action = agent.get_action(state)
            reward= p.act(agent.actions[action])
            next_state = np.reshape(list(p.game.getGameState().values()), [1, agent.state_size])

            # save the sample <s, a, r, s'> to the replay memory
            agent.append_sample(state, action, reward, next_state)
            # every time step do the training
            agent.train_model()
            score += reward
            state = next_state

            if p.game_over():
                # every episode update the target model to be same with model
                agent.update_target_model()

def jeu(nb_frames,agent,p):
    p.reset_game()
    for i_episode in range(nb_frames):
        state = list(p.game.getGameState().values())
        state = np.reshape(state, [1, agent.state_size])
        action= agent.IA(state)
        reward= p.act(agent.actions[action])
        if p.game_over():
            p.reset_game()
            
    
    
def choose_game(path) :
    game_list = []
    for game in os.listdir(path) :
        if (game [:2] != "__" and (game [-2:] == "py")) or (os.path.isdir(path+"/"+game) and 'assets' in os.listdir(path+'/'+ game))  :
            game_list.append(game[:-3] if game [-2:] == "py" else game)
    return game_list

def main():

    
    
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
    DOSSIER = "ressources_network/" + game_name + "_texts"
    FICHIER_NETWORK = DOSSIER + "/" + game_name + "_network.pkl"
    FICHIER_STATE = DOSSIER + "/" + game_name + "_states.txt"
    try:
        os.mkdir(DOSSIER)
    except:
        pass

    p=PLE(game,force_fps=True,display_screen=False)
    agent=essai.DQNAgent(p)  
    
    
    continuer = True
    while continuer :
        nb_parties = int(input ("Phase d'entraînement. \nTaille de l'apprentissage (en parties)?"))
        reward = 0
        p.force_fps = True
        p.display_screen = False
        p.reset_game()
        print ("Début de la phase d'apprentissage.")
        train(nb_parties,agent,p)
        print("Fin de la phase d'apprentissage.")
        
        test = input ("Tester l'IA ? (y/n)")
        if test == "y" :
            nb_frames = 500
            p.force_fps = False
            p.display_screen = True
            jeu(nb_frames,agent,p)
        
        continuer = True if input("Continuer l'apprentissage ? (y/n)") == "y" else False
    
    with open(FICHIER_STATE,'w') as Fichier:
        for state in agent.states:
            Fichier.write(state + ' state\n')
        for i in range(len(agent.actions)):
            Fichier.write(str(agent.actions[i]) +'\n')
    

main()

print("Done")
