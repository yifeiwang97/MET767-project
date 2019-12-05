import torch
import cv2
import sys
import GluttonousSnake as game
from my_dqn_snake import DQN
import numpy as np


# preprocess raw image to 80*80 gray image
def preprocess(observation):
	observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY
	ret, observation = cv2.threshold(observation,127,255,cv2.THRESH_BINARY)
	return np.reshape(observation,(1,1,80,80))

def playSnake():
	# Step 1: init BrainDQN
	#actions = 4
	top = 0
	brain = DQN()
	# Step 2: init Plane Game
	GluttonousSnake = game.GameState()
	# Step 3: play game
	# Step 3.1: obtain init state
	action0 = 1  # [1,0,0]do nothing,[0,1,0]left,[0,0,1]right
	observation0, reward0, terminal,score = GluttonousSnake.frame_step(action0)
	observation0 = preprocess(observation0)
	# cv2.imwrite("3.jpg",observation0,[int(cv2.IMWRITE_JPEG_QUALITY),5])

	brain.state = torch.from_numpy(observation0).type("torch.FloatTensor").cuda()
	brain.load_network()
	step = 0;
	total_step = 100000;
	# Step 3.2: run the game
	while step < total_step:
		action = brain.get_action()
		nextObservation,reward,terminal,score = GluttonousSnake.frame_step(action)
		nextObservation = torch.from_numpy(preprocess(nextObservation)).type("torch.FloatTensor").cuda()
		brain.interface(nextObservation, reward, terminal)
		if score > top:
			top = score
		print('top:%u' , top)

		step = step + 1
	brain.save_network()

def main():
	playSnake()

if __name__ == '__main__':
	main()