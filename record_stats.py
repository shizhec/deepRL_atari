import tensorflow as tf
import numpy as np

class RecordStats:

	def __init__(self, args, test):

		self.test = test
		self.reward = 0
		self.step_count = 0
		self.loss = 0.0
		self.loss_count = 0
		self.games = 0
		self.q_values = 0.0
		self.q_count = 0


		with tf.device('/cpu:0'):
			self.spg = tf.placeholder(tf.float32, shape=[], name="score_per_game")
			self.mean_q = tf.placeholder(tf.float32, shape=[])
			self.total_gp = tf.placeholder(tf.float32, shape=[])

			self.spg_summ = tf.scalar_summary('score_per_game', self.spg)
			self.q_summ = tf.scalar_summary('q_values', self.mean_q)
			self.gp_summ = tf.scalar_summary('steps_per_game', self.total_gp)

			if not test:
				self.mean_l = tf.placeholder(tf.float32, shape=[], name='loss')
				self.l_summ = tf.scalar_summary('loss', self.mean_l)
				self.summary_op = tf.merge_summary([self.spg_summ, self.q_summ, self.gp_summ, self.l_summ])
				self.path = ('records/' + args.game + '/' + args.agent_type + '/' + args.agent_name + '/train')
			else:
				self.summary_op = tf.merge_summary([self.spg_summ, self.q_summ, self.gp_summ])
				self.path = ('records/' + args.game + '/' + args.agent_type + '/' + args.agent_name + '/test')

			# self.summary_op = tf.merge_all_summaries()
			self.sess = tf.Session()
			self.summary_writer = tf.train.SummaryWriter(self.path)

	def record(self, epoch):
		avg_loss = 0
		if self.loss_count != 0:
			avg_loss = self.loss / self.loss_count

		mean_q_values = 0
		if self.q_count > 0:
			mean_q_values = self.q_values / self.q_count

		score_per_game = 0.0
		steps_per_game = 0

		if self.games == 0:
			score_per_game = self.reward
			steps_per_game = self.step_count
		else:
			score_per_game = self.reward / self.games
			steps_per_game = self.step_count / self.games

		score_per_game = float(score_per_game)

		if not self.test:
			summary_str = self.sess.run(self.summary_op, 
				feed_dict={self.spg:score_per_game, self.mean_l:avg_loss, self.mean_q:mean_q_values, self.total_gp:steps_per_game})
			self.summary_writer.add_summary(summary_str, global_step=epoch)
		else:
			summary_str = self.sess.run(self.summary_op, 
				feed_dict={self.spg:score_per_game, self.mean_q:mean_q_values, self.total_gp:steps_per_game})
			self.summary_writer.add_summary(summary_str, global_step=epoch)

		self.reward = 0
		self.step_count = 0
		self.loss = 0
		self.loss_count = 0
		self.games = 0
		self.q_values = 0
		self.q_count = 0


	def add_reward(self, r):
		self.reward += r
		self.step_count += 1

	def add_loss(self, l):
		self.loss += l
		self.loss_count += 1

	def add_game(self):
		self.games += 1

	def add_q_values(self, q_vals):
		mean_q = np.mean(q_vals)
		self.q_values += mean_q
		self.q_count += 1
				