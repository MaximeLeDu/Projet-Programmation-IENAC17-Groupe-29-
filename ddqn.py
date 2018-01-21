import sys
import gym
import pylab
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential


class DQNAgent:
    def __init__(self, p):

        # On récupère les états et les actions, ainsi que leurs tailles
        self.states = p.game.getGameState()
        self.state_size = len(self.states)
        self.actions = p.getActionSet()
        self.action_size = len(self.actions)

        # Paramètres du réseau
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 1000
        # Mémoire
        self.memory = deque(maxlen=2000)

        # Deux réseaux de neurones différents sont utilisés dans cet algorithme
        self.model = self.build_model()
        self.target_model = self.build_model()

        self.update_target_model()


    # Le réseau contient deux couches cachées avec des fonctions d'activation de type ReLU
    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=len(self.states), activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(24, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(len(self.actions), activation='linear',
                        kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # On fixe les poids du réseau cible au réseau courant
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # On récupère l'action suivant une politique epsilon-greedy
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(len(self.actions))
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])
            
    def IA(self,state):
        q_value = self.model.predict(state)
        return np.argmax(q_value[0])

    # On ajoute l'échantillon à la mémoire
    def append_sample(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # On récupère divers échantillons pour l'entraînement
    def train_model(self):
        if len(self.memory) < self.train_start:
            return
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        update_input = np.zeros((batch_size, len(self.states)))
        update_target = np.zeros((batch_size, len(self.states)))
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]

        target = self.model.predict(update_input)
        target_val = self.target_model.predict(update_target)

        for i in range(self.batch_size):
            # On applique l'équation du Q-Learning
            target[i][action[i]] = reward[i] + self.discount_factor * (
                np.amax(target_val[i]))

        self.model.fit(update_input, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)

