
import random
import numpy as np

class DQNAgent():

	def __init__(self, args, q_network, emulator, experience_memory, num_actions, train_stats, test_stats):

		self.network = q_network
		self.emulator = emulator
		self.memory = experience_memory

		self.num_actions = num_actions
		self.history_length = args.history_length
		self.training_frequency = args.training_frequency
		self.random_exploration_length = args.random_exploration_length
		self.initial_exploration_rate = args.initial_exploration_rate
		self.final_exploration_rate = args.final_exploration_rate
		self.final_exploration_frame = args.final_exploration_frame
		self.test_exploration_rate = args.test_exploration_rate
		self.target_update_frequency = args.target_update_frequency
		self.recording_frequency = args.recording_frequency

		self.exploration_rate = self.initial_exploration_rate
		self.total_steps = 0

		self.test_state = []

		if not (test_stats is None):
			self.train_stats = train_stats
			self.test_stats = test_stats
		else:
			self.test_stats = None


	def choose_action(self, obs, epsilon, stats):

		if random.random() >= epsilon:
			if obs is None:
				obs = self.memory.get_current_state()
			q_values = self.network.inference(obs)
			if not (stats is None):
				stats.add_q_values(q_values)
			return np.argmax(q_values)
		else:
			return random.randrange(self.num_actions)


	def checkGameOver(self):
		if self.emulator.isGameOver():
			initial_state = self.emulator.reset()
			for experience in initial_state:
				self.memory.add(experience[0], experience[1], experience[2], experience[3])
			self.train_stats.add_game()


	def run_random_exploration(self):

		for step in range(self.random_exploration_length):

			state, action, reward, terminal, raw_reward = self.act(None, 1.0)
			self.train_stats.add_reward(raw_reward)
			self.memory.add(state, action, reward, terminal)
			self.checkGameOver()
			self.total_steps += 1
			if (self.total_steps % self.recording_frequency == 0) and (not self.total_steps == self.random_exploration_length):
				self.train_stats.record(self.total_steps)


	def act(self, obs, exploration_rate):
		action = self.choose_action(obs, exploration_rate, self.train_stats)
		return self.emulator.run_step(action)


	def run_epoch(self, steps, epoch):

		for step in range(steps):

			state, action, reward, terminal, raw_reward = self.act(None, self.exploration_rate)
			self.memory.add(state, action, reward, terminal)
			self.train_stats.add_reward(raw_reward)
			self.checkGameOver()

			if self.total_steps % self.training_frequency == 0:
				batch = self.memory.get_batch()
				loss = self.network.train(batch[0], batch[1], batch[2], batch[3], batch[4])
				self.train_stats.add_loss(loss)

			if self.total_steps % self.target_update_frequency == 0:
				self.network.update_target_network()

			if self.total_steps < self.final_exploration_frame:
				self.exploration_rate -= (self.exploration_rate - self.final_exploration_rate) / (self.final_exploration_frame - self.total_steps)

			self.total_steps += 1

			if self.total_steps % self.recording_frequency == 0:
				self.train_stats.record(self.total_steps)

		# self.train_stats.record(self.total_steps)
		self.network.save_model(epoch)


	def test_step(self, observation):

		if len(self.test_state) < self.history_length:
			self.test_state.append(observation)

		state = np.expand_dims(np.transpose(self.test_state, [1,2,0]), axis=0)
		action = self.choose_action(state, self.test_exploration_rate, self.test_stats)
		self.test_state.pop(0)
		return action 