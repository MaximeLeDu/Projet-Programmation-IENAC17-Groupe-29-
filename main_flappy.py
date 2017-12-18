import numpy as np
import sys
import pickle

from random import random
from ple import PLE
from ple.games.flappybird import FlappyBird

try:

    DISCRETIZATION = int(sys.argv[1])
    RANDOM_PARADIGM = float(sys.argv[2])
    ALPHA = float(sys.argv[3])
    GAMMA = float(sys.argv[4])
    NORMALIZE = int(sys.argv[5])
    TRAINING = sys.argv[6]
    LENGTH = int(sys.argv[7])
    FICHIER_REWARD = sys.argv[8]
    FICHIER_STATE = sys.argv[9]

except IndexError:
    DISCRETIZATION = 20
    LENGTH = 500
    TRAINING = "play"
    FICHIER_REWARD = "ple/flappy_texts/flappy_rewards.p"
    FICHIER_STATE = "ple/flappy_texts/flappy_states.p"
	
STATES_CHOSEN = []

for i in range(10,len(sys.argv)):
    STATES_CHOSEN.append(sys.argv[i])
	

class Agent():
    def __init__(self, actions, states,fichier_reward,fichier_state):
        self.actions = []
        self.last_action = None
        self.last_states = None
        self.limites = {}
        self.id_states = []
		
        try:
            self.rewards = pickle.load(open(fichier_reward,"rb"))

            with open(fichier_state) as fichier:
                for line in fichier:
                    words=line.strip().split()
                    if len(words)==3:
                        self.limites[words[0]] = [ float(words[1]), float(words[2])]
                        self.id_states.append(words[0])
                    else:
                        if words[0] == 'None':
                            self.actions.append(None)
                        else:
                            self.actions.append(int(words[0]))
                        
            if self.limites == {} or self.actions == []:
                raise AttributeError
                
            print("Les fichiers ont pu être lus")
            
        except Exception as error:
        
            print("Les fichiers n'ont pas pu être lus")
            print(error)
            
            shape=()
			
            if STATES_CHOSEN == []:
                for key in states.keys():
                    self.limites[key] = [0,0]
                    self.id_states.append(key)
                    shape+=(DISCRETIZATION+1,)
                    
            else:
                for state in STATES_CHOSEN:
                    self.limites[state] = [0,0]
                    self.id_states.append(state)
                    shape+=(DISCRETIZATION+1,)
            shape+=(len(actions),)
            print(shape)
            self.rewards=np.zeros(shape)
            self.actions = actions
			    
			    
    def limite_defineur(self,states):
    
        for state in self.id_states:
            if states[state]>self.limites[state][1]:
                self.limites[state][1] = states[state]
            
            if states[state]<self.limites[state][0]:
                self.limites[state][0] = states[state]
            
            
    def random(self):
    
        action = np.random.randint(0, len(self.actions))
        return self.actions[action]
                
    
    def training(self, reward, states):

        if self.last_action!=None:
            choix = random()

			#AI paradigm of choice
            if choix > RANDOM_PARADIGM:
		
                matrix=self.rewards[states]
                action = np.random.choice([action for action, value in enumerate(matrix) if value == np.max(matrix)])

            else:
                action = np.random.randint(0, len(self.actions))
            self.rewards[self.last_states + (self.last_action,)]+=ALPHA*(reward*NORMALIZE + GAMMA*np.max(self.rewards[states]) - self.rewards[self.last_states + (self.last_action,)])
			
		#Initialization 
        else:
            action = np.random.randint(0, len(self.actions))
			
		#Update of the last movements and states
        self.last_states = states
        self.last_action = action
        return self.actions[action]
		
		
    def AI(self,reward,states):
		
        matrix=self.rewards[states]
        print("actions : ", matrix)
        print("states :", states)
        action = np.random.choice([action for action, value in enumerate(matrix) if value == np.max(matrix)])
        return self.actions[action]
		
		
    def discretize(self,states):
    
        self.limite_defineur(states)
        discretized_states=()
        for i in range(len(self.id_states)):
            state = self.id_states[i]
            state_range = self.limites[state][1] - self.limites[state][0]
            state_discretized = int(DISCRETIZATION*(states[state]-self.limites[state][0])/state_range)
            discretized_states+=(state_discretized,)
        return discretized_states
            

game=FlappyBird()

frame_skip = 2
num_steps = 1

if TRAINING == "play":
    force_fps = False  # slower speed
    display_screen = True
    fps = 30
else:
    force_fps = True
    display_screen = False
    if game.allowed_fps != None:
        fps = game.allowed_fps
    else:
        fps = 120

reward = 0.0
max_noops = 20
nb_frames = LENGTH


p = PLE(game, fps=fps, frame_skip=frame_skip, num_steps=num_steps,
        force_fps=force_fps, display_screen=display_screen)

print("creating agent...")
agent = Agent(p.getActionSet(),game.getGameState(),FICHIER_REWARD,FICHIER_STATE)
print("agent created")

print('Pourcentage de complétion : 0%\n')
pourcent =0
# start our training loop
for f in range(nb_frames):
    # if the game is over
    if p.game_over():
        p.reset_game()

    states = game.getGameState()
    if TRAINING == "limits" :
        agent.limite_defineur(states)
        action = agent.random()
    elif TRAINING == "training":
        action = agent.training(reward, agent.discretize(states))
    else:
        action = agent.AI(reward,agent.discretize(states))
    reward = p.act(action)
    if int(f/nb_frames*100)==5+pourcent :
        pourcent = int(f/nb_frames*100)
        print("Pourcentage de complétion : ",pourcent,'%\n')

if TRAINING != 'play':
    print(agent.rewards)
    pickle.dump(agent.rewards,open(FICHIER_REWARD,"wb"))
    with open(FICHIER_STATE,'w') as Fichier:
        for i in range(len(agent.id_states)):
            state = agent.id_states[i]
            Fichier.write(state + ' ' + str(agent.limites[state][0]) + ' ' + str(agent.limites[state][1]) + '\n')
        for i in range(len(agent.actions)):
            Fichier.write(str(agent.actions[i]) +'\n')
