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

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self,first_layer,last_layer):
        super(DQN, self).__init__()
        self.lin1 = nn.Linear(first_layer,2*(first_layer+last_layer))
        self.lin2 = nn.Linear(2*(first_layer+last_layer),first_layer - last_layer)
        self.lin3 = nn.Linear(first_layer - last_layer,last_layer)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        return self.lin3(x)
        
class Agent():

    def __init__(self,actions,states,fichier_network,fichier_state):
        self.actions=[]
        self.states=[]
        self.BATCH_SIZE = 128
        self.GAMMA = 0.999
        self.EPS_START = 0.9
        self.EPS_END = 0.05
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
        self.optimizer = optim.RMSprop(self.net.parameters())


    def select_action(self,state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.step / self.EPS_DECAY)
        self.step+=1
        if sample > eps_threshold:
            rewards = self.net(Variable(state,volatile = True).type(self.FloatTensor)).data
            index = rewards.max(0)[1]
            return index
        else:
            index = self.LongTensor([[random.randrange(len(self.actions))]])
            print(index)
            return index


    def optimize_model():
        global last_sync
        if len(memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)))

        # We don't want to backprop through the expected action values and volatile
        # will save us on temporarily changing the model parameters'
        # requires_grad to False!
        non_final_next_states = Variable(torch.cat([s for s in batch.next_state
                                                    if s is not None]),
                                         volatile=True)
        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = model(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = Variable(torch.zeros(BATCH_SIZE).type(Tensor))
        next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0]
        # Now, we don't want to mess up the loss with a volatile flag, so let's
        # clear it. After this, we'll just end up with a Variable that has
        # requires_grad=False
        next_state_values.volatile = False
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in model.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()
