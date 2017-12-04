import numpy as np
import sys

from random import random
from ple import PLE,matrice_coordonnées
from ple.games.pong import Pong

try:

	DISCRETIZATION = int(sys.argv[1])
	RANDOM_PARADIGM = float(sys.argv[2])
	ALPHA = float(sys.argv[3])
	GAMMA = float(sys.argv[4])
	TRAINING = int(sys.argv[5])
	LENGTH = int(sys.argv[6])

except IndexError:
	DISCRETIZATION=10
	RANDOM_PARADIGM=0.6
	ALPHA = 0.6
	GAMMA = 0.2
	TRAINING = 0
	LENGTH = 50000
	

class Agent():
    def __init__(self, actions, states,nom_fichier):
        self.actions = actions
        self.last_action = 0
        self.last_state = None
        self.limites = {}
        self.id_states = {}
		
        try:
            self.rewards = matrice_coordonnées.uni_to_multi ( matrice_coordonnées.fichier_to_uni (nom_fichier),states)
            with open(nom_fichier) as fichier:
                for line in fichier:
                    try:
                        words=line.strip().split()
                        self.limites[words[0]] = [ int(words[1]), int(words[2])]
                        self.id_states[words[3]] = words[0]
                    except:
                        pass
            print("Le fichier a pu être lu")
        except:
            print("Le fichier n'a pas pu être lu")
            shape=()
            for i in states:
                shape+= (DISCRETIZATION+1,)
            shape+=(len(actions),)
            self.rewards=np.zeros(shape)
			
            i=0
			
            for key in states.keys():
                self.limites[key] = [0,0]
                self.id_states[i] = key
                i+=1
			    
			    
    def limite_defineur(self,state):
        for key in state.keys():
            if state[key]>self.limites[key][1]:
                self.limites[key][1] == state[key]
            
            if state[key]<self.limites[key][0]:
                self.limites[key][0] == state[key]
                
    
    def training(self, reward, state):

        if self.last_action!=0:
            choix = random()

			#AI paradigm of choice
            if choix > RANDOM_PARADIGM:
		
                matrix=self.rewards[state]
                action = np.random.choice([action for action, value in enumerate(matrix) if value == np.max(matrix)])

            else:
                action = np.random.randint(0, len(self.actions))
            self.rewards[self.last_state + (self.last_action,)]+=ALPHA*(reward + GAMMA*self.rewards[state+(action,)] - self.rewards[self.last_state + (self.last_action,)])
			
		#Initialization
        else:
            action = np.random.randint(0, len(self.actions))
			
		#Update of the last movements and states
        self.last_state = state
        self.last_action = action
        return self.actions[action]
		
    def AI(self,reward,state):
		
        matrix=self.rewards[state]
        action = np.random.choice([action for action, value in enumerate(matrix) if value == np.max(matrix)])
        return self.actions[action]
		
    def discretize(self,states):
        discretized_state=[]
        for i in range(len(states)):
            key = self.id_states[i]
            if state[key] > self.limites[key][1]:
                discretized_state.append(DISCRETIZATION)
            elif state[key] < self.limites[key][0]:
                discretized_state.append(0)
            else:
                key_range = self.limites[key][1] - self.limites[key][0]
                key_discretized = int(DISCRETIZATION*(state[key]-self.limites[key][0])/key_range)
                discretized_state.append(key_discretized)
        return discretized_state
            

game=Pong()

frame_skip = 2
num_steps = 1

if TRAINING == 2:
    force_fps = False  # slower speed
    display_screen = True
    fps = 30
else:
    force_fps = True
    display_screen = False
    fps = 120

reward = 0.0
max_noops = 20
nb_frames = LENGTH


p = PLE(game, fps=fps, frame_skip=frame_skip, num_steps=num_steps,
        force_fps=force_fps, display_screen=display_screen)



agent = Agent(p.getActionSet(),game.getGameState(),"ple/pong_rewards.txt")

# start our training loop
for f in range(nb_frames):
    # if the game is over
    if p.game_over():
        p.reset_game()

    state = game.getGameState()
    if TRAINING == 1 :
    	agent.limite_defineur(state)
    else:
        
        if TRAINING == 0:
            action = agent.training(reward, agent.discretize(state))
        else:
    	    action = agent.AI(reward,agent.discretize(state))
        reward = p.act(action)
    if f%100 ==0 :
    	print(f)

matrice_coordonnées.uni_to_fichier( matrice_coordonnées.multi_to_uni(agent.rewards),agent,"ple/Pong_reward.txt")

