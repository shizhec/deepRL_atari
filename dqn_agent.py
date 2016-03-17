
import random
import numpy as np

class DQNAgent():

	def __init__(self, q_network, emulator, experience_memory, observation_length, num_actions, training_frequency, 
		random_exploration_length, initial_exploration_rate, final_exploration_rate, final_exploration_frame, testing_exploration_rate, 
		target_update_frequency):

		self.network = q_network
		self.emulator = emulator
		self.memory = experience_memory

		self.num_actions = num_actions
		self.observation_length = observation_length
		self.training_frequency = training_frequency
		self.random_exploration_length = random_exploration_length
		self.initial_exploration_rate = initial_exploration_rate
		self.final_exploration_rate = final_exploration_rate
		self.final_exploration_frame = final_exploration_frame
		self.testing_exploration_rate = testing_exploration_rate
		self.target_update_frequency = target_update_frequency

		self.exploration_rate = initial_exploration_rate
		self.total_steps = 0
		self.wait_for_state = observation_length

		self.test_state = [] # should this be a numpy array?



	def choose_action(self, obs, epsilon):

		action = None
		if self.wait_for_state > 0:
			self.wait_for_state -= 1
			return 0
		elif random.random() >= epsilon:
			if isinstance(obs, str):
				obs = self.memory.get_current_state()
			q_values = self.network.inference(obs)
			return np.argmax(q_values)
		else:
			return random.randrange(self.num_actions)


	def checkGameOver(self):
		if self.emulator.isGameOver():
			if  self.wait_for_state > 0:
				print("Agent lost during start wait.  Decreasing max_start_wait by 1")
				self.emulator.max_start_wait -= 1
			self.emulator.reset()
			self.wait_for_state = self.observation_length


	def run_random_exploration(self):

		for step in range(self.random_exploration_length):

			experience = self.act("", 1.0)
			self.memory.add(experience[0], experience[1], experience[2], experience[3])
			self.checkGameOver()
			self.total_steps += 1


	def act(self, obs, exploration_rate):
		action = self.choose_action(obs, exploration_rate)
		return self.emulator.run_step(action)


	def run_epoch(self, steps):

		for step in range(steps):

			experience = self.act("", self.exploration_rate)
			self.memory.add(experience[0], experience[1], experience[2], experience[3])
			self.checkGameOver()

			if self.total_steps % self.target_update_frequency == 0:
				self.network.update_target_network()
				self.network.save_model(self.total_steps)

			if self.total_steps % self.training_frequency == 0:
				batch = self.memory.get_batch()
				self.network.train(batch[0], batch[1], batch[2], batch[3])

			if self.total_steps < self.final_exploration_frame:
				self.exploration_rate -= (self.exploration_rate - self.final_exploration_rate) / (self.final_exploration_frame - self.total_steps)

			self.total_steps += 1

			#if self.total_steps % 10000 == 0:
				#print("total_steps(x10000): {0}".format(self.total_steps/10000))


	def test_step(self, observation):

		self.test_state.append(observation)
		if len(self.test_state) < self.observation_length:
			return 0
		else:
			state = np.expand_dims(np.transpose(self.test_state, [1,2,0]), axis=0)
			action = self.choose_action(state, self.testing_exploration_rate)
			self.test_state.pop(0)
			return action


	def test_reset(self):
		self.test_state = []