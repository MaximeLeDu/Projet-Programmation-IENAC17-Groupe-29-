import numpy as np
import sys

from random import random
from ple import PLE,matrice_coordonnées
from ple.games.pong import Pong

try:

	DISCRETIZATION = int(sys.argv[1])
	RANDOM_PARADIGM = int(sys.argv[2])
	ALPHA = int(sys.argv[3])
	GAMMA = int(sys.argv[4])
	TRAINING = int(sys.argv[5])
	LENGTH = int(sys.argv[6])

except IndexError:
	DISCRETIZATION=10
	RANDOM_PARADIGM=0.6
	ALPHA = 0.6
	GAMMA = 0.6
	TRAINING = True
	LENGTH = 50000
	

class Agent():
	def __init__(self, actions, states):
		self.actions = actions
		self.last_action = 0
		self.last_state = None
		
		try:
			self.rewards = matrice_coordonnées.uni_to_multi ( matrice_coordonnées.fichier_to_uni ("ple/Pong_reward.txt"))
			print("Le fichier a pu être lu")
		except:
			print("Le fichier n'a pas pu être lu")
			shape=()
			for i in states:
				shape+= (DISCRETIZATION+1,)
			shape+=(len(actions),)
			self.rewards=np.zeros(shape)

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

game=Pong()

fps = 30  # fps we want to run at
frame_skip = 2
num_steps = 1
force_fps = False  # slower speed
display_screen = True

reward = 0.0
max_noops = 20
nb_frames = LENGTH


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
    if TRAINING :
    	action = agent.training(reward, state_discretized)
    else:
    	action = agent.AI(reward,state_discretized)
    reward = p.act(action)
    if f%100 ==0 :
    	print(f)

matrice_coordonnées.uni_to_fichier( matrice_coordonnées.multi_to_uni(agent.rewards),"ple/Pong_reward.txt")
