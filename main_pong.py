import numpy as np
import torch
import sys

from random import random
from ple import PLE
from ple.games.pong import Pong

try:

	DISCRETIZATION = int(sys.argv[1])
	RANDOM_PARADIGM = int(sys.argv[2])
	ALPHA = int(sys.argv[3])
	GAMMA = int(sys.argv[4])

except IndexError:
	DISCRETIZATION=30
	RANDOM_PARADIGM=0.6
	ALPHA = 0.6
	GAMMA = 0.6

class Agent():
	def __init__(self, actions, states):
		self.actions = actions
		self.last_action = 0
		self.last_state = None

		shape=()
		for i in states:
			shape+= (DISCRETIZATION+1,)
		shape+=(len(actions),)
		self.rewards=np.zeros(shape)

	def pickAction(self, reward, state):

		if self.last_action!=0:
			choix = random()

			#AI paradigm of choice
			if choix > RANDOM_PARADIGM:
		
				matrix=self.rewards[state]
				print(matrix,state,matrix.shape)
				np.random.choice([action for action, value in enumerate(matrix) if value == np.max(matrix)])
				print("choix conscient :",action)

			else:
				action = np.random.randint(0, len(self.actions))
				print(action)
			
			print("last state :", self.last_state, " last action :", self.last_action, "state :", state, "action :",action)
			print(self.last_state+(self.last_action,), state+(action,))
			self.rewards[self.last_state + (self.last_action,)]+=ALPHA*(reward + GAMMA*self.rewards[state+(action,)] - self.rewards[self.last_state + (self.last_action,)])
			
		#Initialization
		else:
			action = np.random.randint(0, len(self.actions))
			print("last state :", self.last_state, " last action :", self.last_action, "state :", state, "action :",action)
			
		#Update of the last movements and states
		self.last_state = state
		self.last_action = action
		print("act:",self.last_action)
		return self.actions[action]

game=Pong()

fps = 30  # fps we want to run at
frame_skip = 2
num_steps = 1
force_fps = False  # slower speed
display_screen = True

reward = 0.0
max_noops = 20
nb_frames = 15000


p = PLE(game, fps=fps, frame_skip=frame_skip, num_steps=num_steps,
        force_fps=force_fps, display_screen=display_screen)


def discretize(game):
	state=game.getGameState()
	
	player_movement_area=game.agentPlayer.SCREEN_HEIGHT-game.agentPlayer.rect_height
	player_discretized = int(DISCRETIZATION*(state["player_y"]-game.agentPlayer.rect_height/2)/player_movement_area)
	
	ball_y_movement_area=game.ball.SCREEN_HEIGHT-2*game.ball.radius
	ball_y_discretized=int(DISCRETIZATION*(state["ball_y"]-game.ball.radius)/ball_y_movement_area)
	
	ball_x_movement_area=game.ball.SCREEN_WIDTH+2*game.ball.radius
	ball_x_discretized=int(DISCRETIZATION*(state["ball_x"]+game.ball.radius)/ball_x_movement_area)
	
	
	
	
	
	
	discretized_states=(player_discretized,ball_x_discretized,ball_y_discretized,1,1)
	return discretized_states


agent = Agent(p.getActionSet(),discretize(game))

# start our training loop
for f in range(nb_frames):
    # if the game is over
    if p.game_over():
        p.reset_game()

    state_discretized=discretize(game)
    action = agent.pickAction(reward, state_discretized)
    reward = p.act(action)

