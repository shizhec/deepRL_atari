import tensorflow as tf
import numpy as np

class RecordStats:

	def __init__(self):

		self.reward = 0
		self.loss = 0
		self.games = 0
		self.activations = 0
		self.act_count = 0
		self.step = 0

		with tf.device('/cpu:0'):
			self.total_r = tf.placeholder(tf.float32, shape=[])
			self.total_l = tf.placeholder(tf.float32, shape=[])
			self.mean_a = tf.placeholder(tf.float32, shape=[])
			self.total_gp = tf.placeholder(tf.float32, shape=[])

			self.r_summ = tf.scalar_summary('reward', self.total_r)
			self.l_summ = tf.scalar_summary('loss', self.total_l)
			self.a_summ = tf.scalar_summary('mean_activation', self.mean_a)
			self.gp_summ = tf.scalar_summary('games_played', self.total_gp)

			self.score_per_game = self.total_r / self.total_gp
			self.spg_summ = tf.scalar_summary('score_per_game', self.score_per_game)

			self.summary_op = tf.merge_all_summaries()

	def record(self):
		# print("r: {0}".format(self.reward))
		# print("l: {0}".format(self.loss))
		# print("gp: {0}".format(self.games))
		# print("act: {0}".format(self.activations))

		summary_str = self.sess.run(self.summary_op, 
			feed_dict={self.total_r:self.reward, self.total_l:self.loss, self.mean_a:self.activations, self.total_gp:self.games})
		self.summary_writer.add_summary(summary_str, global_step=self.step)

		self.reward = 0
		self.loss = 0
		self.games = 0
		self.activations = 0
		self.act_count = 0


	def add_reward(self, r):
		self.reward += r
		self.step += 1

		if self.step % 10000 == 0:
			self.record()

	def add_loss(self, l):
		self.loss += l

	def add_game(self):
		self.games += 1

	def add_activations(self, acts):
		act = np.mean(acts)
		if self.act_count == 0:
			self.activations = act
		else:
			self.activations = ((self.activations * self.act_count) + act) / (self.act_count+1)
		self.act_count += 1


	def add_sess(self, sess):
		self.sess = sess
		self.summary_writer = tf.train.SummaryWriter('records/pong_test2', graph_def=sess.graph_def)
