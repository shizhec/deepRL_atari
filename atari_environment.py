'''
Class for ale instances to generate experiences and test agents.
Uses DeepMind's preproessing/initialization methods
'''

from ale_python_interface import ALEInterface
import cv2
import random
import numpy as np
import sys

class AtariEnvironment:

	def __init__(self, args, stats):
		''' Initialize Atari environment '''

		# Parameters
		self.buffer_length = args.buffer_length
		self.screen_dims = args.screen_dims
		self.frame_skip = args.frame_skip
		self.blend_method = args.blend_method
		self.reward_processing = args.reward_processing
		self.max_start_wait = args.max_start_wait
		self.start_frames_needed = self.buffer_length - 1 + ((args.history_length - 1) * self.frame_skip)

		#Initialize ALE instance
		self.ale = ALEInterface()
		self.ale.setFloat(b'repeat_action_probability', 0.0)
		if args.watch:
			self.ale.setBool(b'sound', True)
			self.ale.setBool(b'display_screen', True)
		self.ale.loadROM(args.rom_path + '/' + args.game + '.bin')

		self.buffer = np.empty((self.buffer_length, 210, 160))
		self.current = 0
		self.action_set = self.ale.getMinimalActionSet()
		self.lives = self.ale.lives()

		self.stats = stats

		self.reset()


	def get_possible_actions(self):
		''' Return list of possible actions for game '''
		return self.action_set

	def get_screen(self):
		''' Add screen to frame buffer '''
		self.buffer[self.current] = np.squeeze(self.ale.getScreenGrayscale())
		self.current = (self.current + 1) % self.buffer_length


	def reset(self):
		self.ale.reset_game()
		self.lives = self.ale.lives()
		if self.stats != None:
			self.stats.add_game()

		if self.max_start_wait < 0:
			print("ERROR: max start wait decreased beyond 0")
			sys.exit()
		elif self.max_start_wait <= self.start_frames_needed:
			wait = 0
		else:
			wait = random.randint(0, self.max_start_wait - self.start_frames_needed)
		for _ in range(wait):
			self.ale.act(self.action_set[0])

		# Fill frame buffer
		self.get_screen()
		for _ in range(self.buffer_length - 1):
			self.ale.act(self.action_set[0])
			self.get_screen()

		state = [(self.preprocess(), 0, 0, False)]
		for step in range(self.history_length - 1):
			state.append(run_step(0))

		# make sure agent hasn't died yet
		if self.isTerminal():
			print("Agent lost during start wait.  Decreasing max_start_wait by 1")
			self.max_start_wait -= 1
			self.reset()
			return self.get_initial_state

		return state


	def run_step(self, action):
		''' Apply action to game and return next screen and reward '''

		reward = 0
		for step in range(self.frame_skip):
			reward += self.ale.act(self.action_set[action])
			self.get_screen()

		if self.stats != None:
			self.stats.add_reward(reward)

		if self.reward_processing == "clip:":
			reward = np.clip(reward, -1, 1)

		terminal = self.isTerminal()
		self.lives = self.ale.lives()

		return (self.preprocess(), action, reward, terminal)



	def preprocess(self):
		''' Preprocess frame for agent '''

		img = None

		if self.blend_method == "max":
			img = np.amax(self.buffer, axis=0)

		return cv2.resize(img, self.screen_dims, interpolation=cv2.INTER_LINEAR)

	def isTerminal(self):
		return (self.isGameOver() or (self.lives > self.ale.lives()))


	def isGameOver(self):
		return self.ale.game_over()
