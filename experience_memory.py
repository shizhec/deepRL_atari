'''
ExperienceMemory is a class for experience replay.  
It stores experience samples and samples minibatches for training.
'''

import numpy as np
import random


class ExperienceMemory:

	def __init__(self, args, num_actions):
		''' Initialize emtpy experience dataset.

		Args:
			capacity: max number of experiences to store
			history_length: number of frames in agent's history
			batch_size: size of training minibatch
			screen_height: height of game screen
			screen_width: width of game screen
		'''

		# params
		self.capacity = args.memory_capacity
		self.history_length = args.history_length
		self.batch_size = args.batch_size
		self.num_actions = num_actions
		self.screen_dims = args.screen_dims

		# initialize dataset
		self.observations = np.empty((capacity, screen_dims[0], screen_dims[1]), dtype=np.uint8)
		self.actions = np.empty(capacity, dtype=np.uint8)
		self.rewards = np.empty(capacity, dtype=np.integer)
		self.terminals = np.empty(capacity, dtype=np.bool)

		self.size = 0
		self.current = 0


	def add(self, obs, act, reward, terminal):
		''' Add experience to dataset.

		Args:
			obs: single observation frame
			act: action taken
			reward: reward
			terminal: is this a terminal state?
		'''

		self.observations[self.current] = obs
		self.actions[self.current] = act
		self.rewards[self.current] = reward
		self.terminals[self.current] = terminal

		self.current = (self.current + 1) % self.capacity
		if self.size == self.capacity - 1:
			self.size = self.capacity
		else:
			self.size = max(self.size, self.current)


	def get_state(self, indices):
		''' Return the observation sequence that ends at index 

		Args:
			indices: list of last observations in sequences
		'''
		state = np.empty((len(indices), self.screen_dims[0], self.screen_dims[1], self.history_length))
		count = 0

		for index in indices:
			# make sure none but last observation are terminal
			# assert not self.terminals[(index - self.history_length + 1):index].any()

			#if self.size >= self.capacity and len(indices) > 1:
				#print("index: {0}".format(index))
				#print("current {0}".format(self.current))
				#print("size {0}".format(self.size))

				#print("slice: {0}".format([index-self.history_length+1, (index+1)%self.capacity]))
				#print("slice_shape: {0}".format(self.observations[((index - self.history_length + 1)):((index + 1)%self.capacity)].shape))

			frame_slice = np.arange(index - self.history_length + 1, (index + 1))
			frame_slice[-1] = frame_slice[-1] % self.capacity
			state[count] = np.transpose(np.take(self.observations, frame_slice, axis=0), [1,2,0])

			#state[count] = np.transpose(self.observations[((index - self.history_length + 1)):((index + 1)%self.capacity):1], [1,2,0]) # fix this
			count += 1

		#print("State shape: {0}".format(state.shape))

		return state


	def get_current_state(self):
		'''  Return most recent observation sequence '''

		return self.get_state([(self.current-1)%self.capacity])


	def get_batch(self):
		''' Sample minibatch of experiences for training '''

		samples = [] # indices of the end of each sample

		while len(samples) < self.batch_size:

			if self.size < self.capacity:  # make this better
				index = random.randint(self.history_length, self.size-1)
			else:
				# make sure state from index doesn't overlap with current's gap
				index = (self.current + random.randint(self.history_length, self.size-1)) % self.capacity
			# make sure no terminal observations are in the first state
			if self.terminals[(index - self.history_length):index].any():
				continue

			samples.append(index)
		#endwhile
		samples = np.asarray(samples)

		# create batch
		o1 = self.get_state((samples - 1) % self.capacity)
		a = np.eye(self.num_actions)[self.actions[samples]] # convert actions to one-hot matrix
		r = self.rewards[samples]
		o2 = self.get_state(samples)
		return [o1, a, r, o2]