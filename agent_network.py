import math
import random
from collections import namedtuple
from ple import PLE
from ple.games import Pong

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

Transition = namedtuple('transition',('state','action','next_state','reward'))

class ReplayMemory(object):
    """Cette classe permet de sauvegarder des états pour les réutiliser ensuite pour l'entraînement."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Sauvegarde une transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    """Ce réseau de neurones est composée d'une couche cachée connectée par deux fonctions linéaires ainsi qu'une 
    fonction d'activation ReLU classique (qui renvoie la partie positive de chaque élément des tenseurs."""

    def __init__(self,first_layer,last_layer):
        super(DQN, self).__init__()
        self.lin1 = nn.Linear(first_layer,(first_layer+last_layer))
        self.lin3 = nn.Linear(first_layer + last_layer,last_layer)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        return self.lin3(x)
        
class Agent():

    def __init__(self,actions,states,fichier_network,fichier_state,learning_rate = 0.001):
        self.actions=[]
        self.states=[]
        self.BATCH_SIZE = 128
        self.GAMMA = 0.999
        self.EPS_START = 0.9
        self.EPS_END = 0.2
        self.EPS_DECAY = 200
        self.step=0
        self.memory = ReplayMemory(10000)
        
        try:
            self.net = torch.load(fichier_network)
            with open(fichier_state) as fichier:
                for line in fichier:
                    words=line.strip().split()
                    if len(words) == 1:
                        if words[0] == 'None':
                            self.actions.append(None)
                        else:
                            self.actions.append(int(words[0]))
                    else:
                        self.states.append(words[0])
                        
            print("Les fichiers ont pu être lus")
                        
        except Exception as error:
            print("Les fichiers n'ont pas pus être lus")
            print(error)
            self.net = DQN(len(states),len(actions))
            self.actions = actions
            self.states = states
            
        use_cuda = torch.cuda.is_available()
        self.FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
        self.ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
        if use_cuda:
            self.net.cuda()
        self.optimizer = optim.Adam(self.net.parameters(),lr=learning_rate)


    def select_action(self,state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.step / self.EPS_DECAY)
        self.step+=1
        if sample > eps_threshold:
            rewards = self.net(Variable(state,volatile = True).type(self.FloatTensor)).data
            index = torch.max(rewards,0)[1]
            return index
        else:
            index = self.LongTensor([[random.randrange(len(self.actions))]])
            return index
            
    def IA(self,state):
        rewards = self.net(Variable(state,volatile = True).type(self.FloatTensor)).data
        return rewards.max(0)[1]


    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = self.ByteTensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)))
        non_final_next_states = Variable(torch.cat([s for s in batch.next_state
                                                    if s is not None]),
                                         volatile=True)
        non_final_next_states.data.resize_(self.BATCH_SIZE,len(self.states))
        state_batch = Variable(torch.cat(batch.state))
        state_batch.data.resize_(self.BATCH_SIZE,len(self.states))
        action_batch = Variable(torch.cat(batch.action))
        action_batch.data.resize_(self.BATCH_SIZE,1)
        reward_batch = Variable(torch.cat(batch.reward))

        # On calcule les récompenses des états du batch puis on les associe aux actions choisies
        state_action_values = self.net(state_batch).gather(1, action_batch)

        # On calcule les valeurs expérimentales des états suivants
        next_state_values = Variable(torch.zeros(self.BATCH_SIZE).type(self.FloatTensor))
        
        next_state_values[non_final_mask] = self.net(non_final_next_states).max(1)[0]
        # On ne veut pas perturber la rétropropagation par un tensor volatile
        next_state_values.volatile = False
        # Calcule la valeur attendue par le Q-Learning
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch
        # Calcule la perte de Huber
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        # Optimise le modèle
        self.optimizer.zero_grad()
        loss.backward()
        
        self.optimizer.step()
