import tensorflow as tf


class QNetwork():

	def __init__(self, args, num_actions):
		''' Build tensorflow graph for deep q network '''

		self.discount_factor = args.discount_factor
		self.path = 'saved_models/' + args.game + '/' + args.agent_type + '/' + args.agent_name
		self.name = args.agent_name

		# input placeholders
		self.observation = tf.placeholder(tf.float32, shape=[None, args.screen_dims[0], args.screen_dims[1], args.history_length], name="observation")
		self.actions = tf.placeholder(tf.float32, shape=[None, num_actions], name="actions") # one-hot matrix because tf.gather() doesn't support multidimensional indexing yet
		self.rewards = tf.placeholder(tf.float32, shape=[None], name="rewards")
		self.next_observation = tf.placeholder(tf.float32, shape=[None, args.screen_dims[0], args.screen_dims[1], args.history_length], name="next_observation")
		self.terminals = tf.placeholder(tf.float32, shape=[None], name="terminals")

		num_conv_layers = len(args.conv_kernel_shapes)
		assert(num_conv_layers == len(args.conv_strides))
		num_dense_layers = len(args.dense_layer_shapes)

		last_policy_layer = None
		last_target_layer = None
		self.update_target = []
		self.policy_network_params = []

		# initialize convolutional layers
		for layer in range(num_conv_layers):
			policy_input = None
			target_input = None
			if layer == 0:
				policy_input = self.observation
				target_input = self.next_observation
			else:
				policy_input = last_policy_layer
				target_input = last_target_layer

			last_layers = self.conv_relu(policy_input, target_input, 
				args.conv_kernel_shapes[layer], args.conv_strides[layer], layer)
			last_policy_layer = last_layers[0]
			last_target_layer = last_layers[1]

		# initialize fully-connected layers
		for layer in range(num_dense_layers):
			policy_input = None
			target_input = None
			if layer == 0:
				input_size = args.dense_layer_shapes[0][0]
				policy_input = tf.reshape(last_policy_layer, shape=[-1, input_size])
				target_input = tf.reshape(last_target_layer, shape=[-1, input_size])
			else:
				policy_input = last_policy_layer
				target_input = last_target_layer

			last_layers = self.dense_relu(policy_input, target_input, args.dense_layer_shapes[layer], layer)
			last_policy_layer = last_layers[0]
			last_target_layer = last_layers[1]


		# initialize q_layer
		last_layers = self.dense_linear(
			last_policy_layer, last_target_layer, [args.dense_layer_shapes[-1][-1], num_actions])
		self.policy_q_layer = last_layers[0]
		self.target_q_layer = last_layers[1]

		self.loss = self.build_loss(args.error_clipping, num_actions)

		self.train_op = None  # add options for more optimizers
		if args.optimizer == 'rmsprop':
			self.train_op = tf.train.RMSPropOptimizer(
				args.learning_rate, decay=args.rmsprop_decay, momentum=0.0, epsilon=args.rmsprop_epsilon).minimize(self.loss)
		elif args.optimizer == 'graves_rmsprop':
			self.train_op = self.build_rmsprop_optimizer(args.learning_rate, args.rmsprop_decay, args.rmsprop_epsilon)

		self.saver = tf.train.Saver(self.policy_network_params)

		# start tf session
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333333)  # avoid using all vram for GTX 970
		self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

		if args.watch:
			load_path = tf.train.latest_checkpoint(self.path)
			self.saver.restore(self.sess, load_path)		
		else:
			self.sess.run(tf.initialize_all_variables())


	def conv_relu(self, policy_input, target_input, kernel_shape, stride, layer_num):
		''' Build a convolutional layer

		Args:
			input_layer: input to convolutional layer - must be 3d
			target_input: input to layer of target network - must also be 3d
			kernel_shape: tuple for filter shape: (filter_height, filter_width, in_channels, out_channels)
			stride: tuple for stride: (1, vert_stride. horiz_stride, 1)
		'''
		name = 'conv' + str(layer_num + 1)
		with tf.variable_scope(name):

			weights = tf.Variable(tf.truncated_normal(kernel_shape, stddev=0.01), name=(name + "_weights"))
			biases = tf.Variable(tf.fill([kernel_shape[-1]], 0.1), name=(name + "_biases"))

			activation = tf.nn.relu(tf.nn.conv2d(policy_input, weights, stride, 'VALID') + biases)

			target_weights = tf.Variable(weights.initialized_value(), trainable=False, name=("target_" + name + "_weights"))
			target_biases = tf.Variable(biases.initialized_value(), trainable=False, name=("target_" + name + "_biases"))

			target_activation = tf.nn.relu(tf.nn.conv2d(target_input, target_weights, stride, 'VALID') + target_biases)

			self.update_target.append(target_weights.assign(weights))
			self.update_target.append(target_biases.assign(biases))

			self.policy_network_params.append(weights)
			self.policy_network_params.append(biases)

			return [activation, target_activation]


	def dense_relu(self, policy_input, target_input, shape, layer_num):
		''' Build a fully-connected relu layer 

		Args:
			input_layer: input to dense layer
			target_input: input to layer of target network
			shape: tuple for weight shape (num_input_nodes, num_layer_nodes)
		'''
		name = 'dense' + str(layer_num + 1)
		with tf.variable_scope(name):

			weights = tf.Variable(tf.truncated_normal(shape, stddev=0.01), name=(name + "_weights"))
			biases = tf.Variable(tf.fill([shape[-1]], 0.1), name=(name + "_biases"))

			activation = tf.nn.relu(tf.matmul(policy_input, weights) + biases)

			target_weights = tf.Variable(weights.initialized_value(), trainable=False, name=("target_" + name + "_weights"))
			target_biases = tf.Variable(biases.initialized_value(), trainable=False, name=("target_" + name + "_biases"))

			target_activation = tf.nn.relu(tf.matmul(target_input, target_weights) + target_biases)

			self.update_target.append(target_weights.assign(weights))
			self.update_target.append(target_biases.assign(biases))

			self.policy_network_params.append(weights)
			self.policy_network_params.append(biases)

			return [activation, target_activation]


	def dense_linear(self, policy_input, target_input, shape):
		''' Build the fully-connected linear output layer 

		Args:
			input_layer: last hidden layer
			target_input: last hidden layer of target network
			shape: tuple for weight shape (num_input_nodes, num_actions)
		'''
		name = 'q_layer'
		with tf.variable_scope(name):

			weights = tf.Variable(tf.truncated_normal(shape, stddev=0.01), name=(name + "_weights"))
			biases = tf.Variable(tf.fill([shape[-1]], 0.1), name=(name + "_biases"))

			activation = tf.matmul(policy_input, weights) + biases

			target_weights = tf.Variable(weights.initialized_value(), trainable=False, name=("target_" + name + "_weights"))
			target_biases = tf.Variable(biases.initialized_value(), trainable=False, name=("target_" + name + "_biases"))

			target_activation = tf.matmul(target_input, target_weights) + target_biases

			self.update_target.append(target_weights.assign(weights))
			self.update_target.append(target_biases.assign(biases))

			self.policy_network_params.append(weights)
			self.policy_network_params.append(biases)

			return [activation, target_activation]



	def inference(self, obs):
		''' Get state-action value predictions for an observation 

		Args:
			observation: the observation
		'''

		return self.sess.run(self.policy_q_layer, feed_dict={self.observation:obs})


	def build_loss(self, error_clip, num_actions):
		''' build loss graph '''
		with tf.name_scope("loss"):

			predictions = tf.reduce_sum(tf.mul(self.policy_q_layer, self.actions), 1)
			optimality = tf.reduce_max(self.target_q_layer, 1)
			targets = tf.stop_gradient(self.rewards + (self.discount_factor * optimality * self.terminals))

			'''
			# Double Q-Learning:
			max_actions = tf.to_int32(tf.argmax(self.policy_q_layer, 1))
			# tf.gather doesn't support multidimensional indexing yet, so we flatten output activations for indexing
			indices = tf.range(0, tf.size(max_actions) * num_actions, num_actions) + max_actions
			max_action_values = tf.gather(tf.reshape(self.target_q_layer, shape=[-1]), indices)
			targets = tf.stop_gradient(self.rewards + (self.discount_factor * max_action_values))
			'''

			difference = tf.abs(predictions - targets)

			if error_clip >= 0:
				quadratic_part = tf.clip_by_value(difference, 0.0, error_clip)
				linear_part = difference - quadratic_part
				errors = (0.5 * tf.square(quadratic_part)) + (error_clip * linear_part)
			else:
				errors = (0.5 * tf.square(difference))

			return tf.reduce_sum(errors)  # add option for reduce mean?


	def train(self, o1, a, r, o2, t):
		''' train network on batch of experiences

		Args:
			o1: first observations
			a: actions taken
			r: rewards received
			o2: succeeding observations
		'''

		return self.sess.run([self.train_op, self.loss], 
			feed_dict={self.observation:o1, self.actions:a, self.rewards:r, self.next_observation:o2, self.terminals:t})[1]


	def update_target_network(self):
		''' update weights and biases of target network '''

		self.sess.run(self.update_target)


	def save_model(self, epoch):

		self.saver.save(self.sess, self.path + '/' + self.name + '.ckpt', global_step=epoch)


	def build_rmsprop_optimizer(self, learning_rate, rmsprop_decay, rmsprop_constant):

		with tf.name_scope('rmsprop'):
			optimizer = tf.train.GradientDescentOptimizer(learning_rate)

			grads_and_vars = optimizer.compute_gradients(self.loss)
			grads = [gv[0] for gv in grads_and_vars]
			params = [gv[1] for gv in grads_and_vars]

			square_grads = [tf.square(grad) for grad in grads]

			avg_grads = [tf.Variable(tf.ones(var.get_shape())) for var in params]
			avg_square_grads = [tf.Variable(tf.ones(var.get_shape())) for var in params]

			update_avg_grads = [grad_pair[0].assign((rmsprop_decay * grad_pair[0]) + ((1 - rmsprop_decay) * grad_pair[1])) 
				for grad_pair in zip(avg_grads, grads)]
			update_avg_square_grads = [grad_pair[0].assign((rmsprop_decay * grad_pair[0]) + ((1 - rmsprop_decay) * tf.square(grad_pair[1]))) 
				for grad_pair in zip(avg_square_grads, grads)]
			avg_grad_updates = update_avg_grads + update_avg_square_grads

			rms = [tf.abs(tf.sqrt(avg_grad_pair[1] - tf.square(avg_grad_pair[0]) + rmsprop_constant)) 
				for avg_grad_pair in zip(avg_grads, avg_square_grads)]

			rms_updates = [grad_rms_pair[0] / grad_rms_pair[1] for grad_rms_pair in zip(grads, rms)]
			train = optimizer.apply_gradients(zip(rms_updates, params))


			'''
			exp_mov_avg = tf.train.ExponentialMovingAverage(rmsprop_decay)  

			update_avg_grads = exp_mov_avg.apply(grads) # ??? tf bug? Why doesn't this work?
			update_avg_square_grads = exp_mov_avg.apply(square_grads)

			rms = tf.abs(tf.sqrt(exp_mov_avg.average(square_grads) - tf.square(exp_mov_avg.average(grads) + rmsprop_constant)))
			rms_updates = grads / rms
			train = opt.apply_gradients(zip(rms_updates, params))
			'''

			return tf.group(train, *avg_grad_updates)