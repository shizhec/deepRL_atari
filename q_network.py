import tensorflow as tf

# TODO: custom rmsprop
# TODO: clip delta?


class QNetwork():

	def __init__(self, conv_kernel_shapes, conv_strides, dense_layer_shapes, num_actions, observation_length, screen_height, screen_width, 
		discount_factor, learning_rate, rmsprop_decay, rmsprop_constant, stats):
		''' Build tensorflow graph for deep q network

		Args:
			conv_kernel_shapes: list of tuples for shapes of convolutional kernels: (filter_height, filter_width, in_channels, out_channels)
			conv_strides: list of tuples for convolutional strides. ex: (1,2,2,1) for stride 2
			dense_layer_shapes: list of tuples for dense layer shapes: (num_input_nodes, num_nodes)
			num_actions: number of possible actions
			observation_length: number of frames in an observation
			screen_height: height of game screen
			screen width: width of game screen
			discount_factor: constant used to discount future rewards
			learning_rate: constant for magnitude of gradient updates
			rmsprop_decay: constant for updating moving root-mean-square for rmsprop
			rmsprop_constant: constant to avoid division by zero for rmsprop
		'''

		self.discount_factor = tf.constant(discount_factor, name="discount_factor")

		self.stats = stats

		# input placeholders
		self.observation = tf.placeholder(tf.float32, shape=[None, screen_height, screen_width, observation_length], name="observation")
		self.actions = tf.placeholder(tf.float32, shape=[None, num_actions], name="actions") # one-hot matrix because tf.gather() doesn't support multidimensional indexing yet
		self.rewards = tf.placeholder(tf.float32, shape=[None], name="rewards")
		self.next_observation = tf.placeholder(tf.float32, shape=[None, screen_height, screen_width, observation_length], name="next_observation")

		num_conv_layers = len(conv_kernel_shapes)
		assert(num_conv_layers == len(conv_strides))
		num_dense_layers = len(dense_layer_shapes)

		self.conv_weights = []
		self.conv_biases = []
		self.conv_layers = []
		self.dense_weights = []
		self.dense_biases = []
		self.dense_layers = []
		self.target_conv_weights = []
		self.target_conv_biases = []
		self.target_conv_layers = []
		self.target_dense_weights = []
		self.target_dense_biases = []
		self.target_dense_layers = []

		# initialize convolutional layers
		for layer in range(num_conv_layers):
			input_layer = None
			target_layer = None
			if layer == 0:
				input_layer = self.observation
				target_input = self.next_observation
			else:
				input_layer = self.conv_layers[layer-1]
				target_input = self.target_conv_layers[layer-1]

			self.conv_relu(input_layer, target_input, conv_kernel_shapes[layer], conv_strides[layer])

		# initialize fully-connected layers
		for layer in range(num_dense_layers):
			input_layer = None
			target_layer = None
			if layer == 0:
				input_size = dense_layer_shapes[0][0]
				input_layer = tf.reshape(self.conv_layers[-1], shape=[-1, input_size])
				target_input = tf.reshape(self.target_conv_layers[-1], shape=[-1, input_size])
			else:
				input_layer = self.dense_layers[layer-1]
				target_input = self.target_dense_layers[layer-1]

			self.dense_relu(input_layer, target_input, dense_layer_shapes[layer])


		# initialize q_layer
		self.dense_linear(self.dense_layers[-1], self.target_dense_layers[-1], [dense_layer_shapes[-1][-1], num_actions])

		# graph for updating target network
		params = self.conv_weights + self.conv_biases + self.dense_weights + self.dense_biases + [self.q_weights] + [self.q_biases]
		target_params = self.target_conv_weights + self.target_conv_biases + self.target_dense_weights + self.target_dense_biases + [self.target_q_weights] + [self.target_q_biases]

		self.update_target = [target_params[i].assign(params[i]) for i in range(len(params))]

		self.loss = self.build_loss()

		self.train_op = tf.train.RMSPropOptimizer(learning_rate, decay=rmsprop_decay, momentum=0.0, epsilon=rmsprop_constant).minimize(self.loss)

		# start tf session
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)  # avoid using all vram for GTX 970
		self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
		self.sess.run(tf.initialize_all_variables())
		self.stats.add_sess(self.sess)


	def conv_relu(self, input_layer, target_input, kernel_shape, stride):
		''' Build a convolutional layer

		Args:
			input_layer: input to convolutional layer - must be 3d
			target_input: input to layer of target network - must also be 3d
			kernel_shape: tuple for filter shape: (filter_height, filter_width, in_channels, out_channels)
			stride: tuple for stride: (1, vert_stride. horiz_stride, 1)
		'''

		self.conv_weights.append(tf.Variable(tf.truncated_normal(kernel_shape, stddev=0.01), name="conv{0}_weights".format(len(self.conv_weights))))
		self.conv_biases.append(tf.Variable(tf.fill([kernel_shape[-1]], 0.01), name="conv{0}_biases".format(len(self.conv_biases))))

		conv = tf.nn.conv2d(input_layer, self.conv_weights[-1], stride, 'VALID')
		self.conv_layers.append(tf.nn.relu(tf.add(conv, self.conv_biases[-1])))

		self.target_conv_weights.append(tf.Variable(self.conv_weights[-1].initialized_value(), trainable=False, name="target_conv{0}_weights".format(len(self.target_conv_weights))))
		self.target_conv_biases.append(tf.Variable(self.conv_biases[-1].initialized_value(), trainable=False, name="target_conv{0}_biases".format(len(self.target_conv_biases))))

		t_conv = tf.nn.conv2d(target_input, self.target_conv_weights[-1], stride, 'VALID')
		self.target_conv_layers.append(tf.nn.relu(tf.add(t_conv, self.target_conv_biases[-1])))


	def dense_relu(self, input_layer, target_input, shape):
		''' Build a fully-connected relu layer 

		Args:
			input_layer: input to dense layer
			target_input: input to layer of target network
			shape: tuple for weight shape (num_input_nodes, num_layer_nodes)
		'''

		self.dense_weights.append(tf.Variable(tf.truncated_normal(shape, stddev=0.01), name="dense{0}_weights".format(len(self.dense_weights))))
		self.dense_biases.append(tf.Variable(tf.fill([shape[-1]], 0.01), name="dense{0}_biases".format(len(self.dense_biases))))

		weight_sum = tf.matmul(input_layer, self.dense_weights[-1])
		self.dense_layers.append(tf.nn.relu(tf.add(weight_sum, self.dense_biases[-1])))

		self.target_dense_weights.append(tf.Variable(self.dense_weights[-1].initialized_value(), trainable=False, name="target_dense{0}_weights".format(len(self.target_dense_weights))))
		self.target_dense_biases.append(tf.Variable(self.dense_biases[-1].initialized_value(), trainable=False, name="target_dense{0}_biases".format(len(self.target_dense_biases))))

		t_sum = tf.matmul(target_input, self.target_dense_weights[-1])
		self.target_dense_layers.append(tf.nn.relu(tf.add(t_sum, self.target_dense_biases[-1])))


	def dense_linear(self, input_layer, target_input, shape):
		''' Build the fully-connected linear output layer 

		Args:
			input_layer: last hidden layer
			target_input: last hidden layer of target network
			shape: tuple for weight shape (num_input_nodes, num_actions)
		'''

		self.q_weights = tf.Variable(tf.truncated_normal(shape, stddev=0.01), name="q_weights")
		self.q_biases = tf.Variable(tf.fill([shape[-1]], 0.01), name="q_biases")

		self.q_layer = tf.add(tf.matmul(input_layer, self.q_weights), self.q_biases)

		self.target_q_weights = tf.Variable(self.q_weights.initialized_value(), trainable=False, name="target_q_weights")
		self.target_q_biases = tf.Variable(self.q_biases.initialized_value(), trainable=False, name="target_q_biases")

		self.target_q_layer = tf.add(tf.matmul(target_input, self.target_q_weights), self.target_q_biases)


	def inference(self, obs):
		''' Get state-action value predictions for an observation 

		Args:
			observation: the observation
		'''

		q_values =  self.sess.run(self.q_layer, feed_dict={self.observation:obs})
		self.stats.add_activations(q_values)
		return q_values

	def build_loss(self):
		''' build loss graph '''

		predictions = tf.reduce_sum(tf.mul(self.q_layer, self.actions), 1)
		optimality = tf.reduce_max(self.target_q_layer, 1)
		targets = tf.add(self.rewards, tf.mul(self.discount_factor, optimality))
		return tf.reduce_sum(tf.mul(0.5, tf.square(tf.sub(predictions, targets))))


	def train(self, o1, a, r, o2):
		''' train network on batch of experiences

		Args:
			o1: first observations
			a: actions taken
			r: rewards received
			o2: succeeding observations
		'''

		loss = self.sess.run([self.train_op, self.loss], feed_dict={self.observation:o1, self.actions:a, self.rewards:r, self.next_observation:o2})[1]
		self.stats.add_loss(loss)


	def update_target_network(self):
		''' update weights and biases of target network '''

		self.sess.run(self.update_target)