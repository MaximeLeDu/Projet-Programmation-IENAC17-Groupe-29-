import numpy as np
import sys
import pickle

from random import random
from ple import PLE


class Agent():

    def __init__(self, actions, states,fichier_reward,fichier_state,STATES_CHOSEN,NEW_STATES_CHOSEN):
        self.actions = []
        self.last_action = None
        self.last_states = None
        self.limites = {}
        self.id_states = []
        self.DISCRETIZATION = 50
        self.LENGTH = 1000
        self.RANDOM_PARADIGM=0.6
        self.ALPHA = 0.1
        self.GAMMA = 0.9
        self.NORMALIZE = 1
		
        try:
            self.rewards = pickle.load(open(fichier_reward,"rb"))

            with open(fichier_state) as fichier:
                for line in fichier:
                    words=line.strip().split()
                    if len(words)==3:
                        self.limites[words[0]] = [ float(words[1]), float(words[2])]
                        self.id_states.append(words[0])
                    elif len(words)==2:
                        self.DISCRETIZATION=words[1]
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
            
            states = NEW_STATES_CHOSEN + STATES_CHOSEN
            shape=()
			
            if states == []:
                for key in states.keys():
                    self.limites[key] = [0,0]
                    self.id_states.append(key)
                    shape+=(self.DISCRETIZATION+1,)
                    
            else:
                for state in states:
                    self.limites[state] = [0,0]
                    self.id_states.append(state)
                    shape+=(self.DISCRETIZATION+1,)
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
            
            
    def new_states_add(self,new_states):
        previous_len = len(self.id_states)
        for state in new_states :
            self.limites[state] = [0,0]
            self.id_states.append(state)
            
            
        len_tot = len(self.id_states)    
        shape=()
        index =(0,)
        tot_elem=1
            
        for i in range(len_tot) :
            shape+=(self.DISCRETIZATION+1,)
            index+=(0,)
            tot_elem*=(self.DISCRETIZATION+1)
        shape+=(len(self.actions),)
        tot_elem*=len(self.actions)
        
        new_matrix = np.zeros(shape)
        

        index = list(index)
        
        for i in range (tot_elem-1) :
            new_matrix[tuple(index)] = self.rewards[tuple(index)[:previous_len]+(index[-1],)]
            self.coord_increment(index,shape)
        new_matrix[tuple(index)] = self.rewards[tuple(index)[:previous_len]+(index[-1],)]
        self.rewards=new_matrix

    
    def coord_increment(self,coord,shape,i=0) :
        if coord[i] == shape[i]-1 :
            coord[i]=0
            self.coord_increment(coord,shape,i+1)
        else :
            coord[i]+=1
    

            
            
    def random(self):
    
        action = np.random.randint(0, len(self.actions))
        return self.actions[action]
                
    
    def training(self, reward, states):

        if self.last_action!=None:
            choix = random()

			#AI paradigm of choice
            if choix > self.RANDOM_PARADIGM:
		
                matrix=self.rewards[states]
                action = np.random.choice([action for action, value in enumerate(matrix) if value == np.max(matrix)])

            else:
                action = np.random.randint(0, len(self.actions))
            self.rewards[self.last_states + (self.last_action,)]+=self.ALPHA*(reward*self.NORMALIZE + self.GAMMA*np.max(self.rewards[states]) - self.rewards[self.last_states + (self.last_action,)])
			
		#Initialization 
        else:
            action = np.random.randint(0, len(self.actions))
			
		#Update of the last movements and states
        self.last_states = states
        self.last_action = action
        return self.actions[action]
		
		
    def AI(self,reward,states):
		
        matrix=self.rewards[states]
        """
        print("actions : ", matrix)
        print("states :", states)
        """
        action = np.random.choice([action for action, value in enumerate(matrix) if value == np.max(matrix)])
        return self.actions[action]
		
		
    def discretize(self,states):
    
        self.limite_defineur(states)
        discretized_states=()
        for i in range(len(self.id_states)):
            state = self.id_states[i]
            state_range = self.limites[state][1] - self.limites[state][0]
            state_discretized = int(self.DISCRETIZATION*(states[state]-self.limites[state][0])/state_range)
            discretized_states+=(state_discretized,)
        return discretized_states