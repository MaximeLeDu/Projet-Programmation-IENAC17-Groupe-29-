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
	TRAINING = sys.argv[5]
	LENGTH = int(sys.argv[6])

except IndexError:
	DISCRETIZATION=10
	RANDOM_PARADIGM=0.6
	ALPHA = 0.1
	GAMMA = 0.9
	TRAINING = "limits"
	LENGTH = 500
	

class Agent():
    def __init__(self, actions, states,fichier_reward,fichier_state):
        self.actions = []
        self.last_action = None
        self.last_state = None
        self.limites = {}
        self.id_states = {}
        self.id_actions = {}
		
        try:
            self.rewards = matrice_coordonnées.uni_to_multi ( matrice_coordonnées.fichier_to_uni (fichier_reward))
            with open(fichier_state) as fichier:
                for line in fichier:
                    words=line.strip().split()
                    if len(words)==4:
                        self.limites[words[0]] = [ float(words[1]), float(words[2])]
                        self.id_states[int(words[3])] = words[0]
                    else:
                        if words[0] == 'None':
                            self.actions.append(None)
                        else:
                            self.actions.append(int(words[0]))
                        
            if self.limites == {} or self.id_actions == []:
                raise AttributeError
                
            print("Les fichiers ont pu être lus")
            
        except Exception as error:
        
            print("Les fichiers n'ont pas pu être lus")
            print(error)
            
            shape=()
            for i in states:
                shape+= (DISCRETIZATION+1,)
            shape+=(len(actions),)
            self.rewards=np.zeros(shape)
            self.actions = actions
			
            i=0
			
            for key in states.keys():
                self.limites[key] = [0,0]
                self.id_states[i] = key
                i+=1
                
            i=0
            
            for action in actions:
                self.id_actions[i] = action
			    
			    
    def limite_defineur(self,state):
    
        for key in state.keys():
            if state[key]>self.limites[key][1]:
                self.limites[key][1] = state[key]
            
            if state[key]<self.limites[key][0]:
                self.limites[key][0] = state[key]
            
            
    def random(self):
    
        action = np.random.randint(0, len(self.actions))
        return self.actions[action]
                
    
    def training(self, reward, state):

        if self.last_action!=None:
            choix = random()

			#AI paradigm of choice
            if choix > RANDOM_PARADIGM:
		
                matrix=self.rewards[state]
                action = np.random.choice([action for action, value in enumerate(matrix) if value == np.max(matrix)])

            else:
                action = np.random.randint(0, len(self.actions))
            self.rewards[self.last_state + (self.last_action,)]+=ALPHA*(reward + GAMMA*np.max(self.rewards[state]) - self.rewards[self.last_state + (self.last_action,)])
            print("self.rewards[ ", self.last_state + (self.last_action,), " ] = ", self.rewards[self.last_state + (self.last_action,)])
			
		#Initialization 
        else:
            action = np.random.randint(0, len(self.actions))
			
		#Update of the last movements and states
        self.last_state = state
        self.last_action = action
        return self.actions[action]
		
		
    def AI(self,reward,state):
		
        matrix=self.rewards[state]
        print('shape matrix : ', matrix.shape)
        print('reward.shape : ', self.rewards.shape)
        print('state : ', state)
        action = np.random.choice([action for action, value in enumerate(matrix) if value == np.max(matrix)])
        return self.actions[action]
		
		
    def discretize(self,states):
    
        self.limite_defineur(states)
        discretized_state=()
        for i in range(len(states)):
            key = self.id_states[i]
            key_range = self.limites[key][1] - self.limites[key][0]
            key_discretized = int(DISCRETIZATION*(state[key]-self.limites[key][0])/key_range)
            discretized_state+=(key_discretized,)
        return discretized_state
            

game=Pong()

frame_skip = 2
num_steps = 1

if TRAINING == "play":
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



agent = Agent(p.getActionSet(),game.getGameState(),"ple/pong_rewards.txt","ple/pong_states.txt")
# start our training loop
for f in range(nb_frames):
    # if the game is over
    if p.game_over():
        p.reset_game()

    state = game.getGameState()
    if TRAINING == "limits" :
    	agent.limite_defineur(state)
    	action = agent.random()
    elif TRAINING == "training":
        action = agent.training(reward, agent.discretize(state))
    else:
    	action = agent.AI(reward,agent.discretize(state))
    reward = p.act(action)
    if f%10000 ==0 :
    	print(f)

matrice_coordonnées.uni_to_fichier( matrice_coordonnées.multi_to_uni(agent.rewards),"ple/pong_rewards.txt")
matrice_coordonnées.save_state("ple/pong_states.txt",agent)
