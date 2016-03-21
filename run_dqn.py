import sys
import argparse
from atari_emulator import AtariEmulator
from experience_memory import ExperienceMemory
from q_network import QNetwork
from dqn_agent import DQNAgent
from record_stats import RecordStats
import experiment

def main():

	parser = argparse.ArgumentParser('a program to train or run a deep q-learning agent')
	parser.add_argument("name", type=str, help="unique name of this agent")
	parser.add_argument("game", type=str, help="name of game to play")
	parser.add_argument("--rom_path", type=str, help="path to directory containing atari game roms", default='roms')
	parser.add_argument("--watch",
		help="if true, a pretrained model with the specified name is loaded and tested with the game screen displayed", 
		action='store_true')

	parser.add_argument("--epochs", type=int, help="number of epochs", default=200)
	parser.add_argument("--epoch_length", type=int, help="number of steps in an epoch", default=250000)
	parser.add_argument("--test_steps", type=int, help="max number of steps per test", default=125000)
	parser.add_argument("--test_episodes", type=int, help="max number of episodes per test", default=30)
	parser.add_argument("--history_length", type=int, help="number of frames in a state", default=4)
	parser.add_argument("--training_frequency", type=int, help="number of steps run before training", default=4)
	parser.add_argument("--random_exploration_length", type=int, 
		help="number of randomly-generated experiences to initially fill experience memory", default=50000)
	parser.add_argument("--initial_exploration_rate", type=float, help="initial exploration rate", default=1.0)
	parser.add_argument("--final_exploration_rate", type=float, help="final exploration rate from linear annealing", default=0.1)
	parser.add_argument("--final_exploration_frame", type=int, 
		help="frame at which the final exploration rate is reached", default=1000000)
	parser.add_argument("--test_exploration_rate", type=float, help="exploration rate while testing", default=0.05)
	parser.add_argument("--frame_skip", type=int, help="number of frames to repeat chosen action", default=4)
	parser.add_argument("--screen_dims", type=tuple, help="dimensions to resize frames", default=(84,84))
	# used for stochasticity and to help prevent overfitting.  
	# Must be greater than frame_skip * (observation_length -1) + buffer_length - 1
	parser.add_argument("--max_start_wait", type=int, help="max number of steps to wait for initial state", default=30)
	# buffer_length = 1 prevents blending
	parser.add_argument("--buffer_length", type=int, help="length of buffer to blend frames", default=2)
	parser.add_argument("--blend_method", type=str, help="method used to blend frames", choices=('max'), default='max')
	parser.add_argument("--reward_processing", type=str, help="method to process rewards", choices=('clip', 'none'), default='clip')
	# must set network_architecture to custom in order use custom architecture
	parser.add_argument("--conv_kernel_shapes", type=tuple, 
		help="shapes of convnet kernels: ((height, width, in_channels, out_channels), (next layer))")
	# must have same length as conv_kernel_shapes
	parser.add_argument("--conv_strides", type=tuple, help="connvet strides: ((1, height, width, 1), (next layer))")
	# currently,  you must have at least one dense layer
	parser.add_argument("--dense_layer_shapes", type=tuple, help="shapes of dense layers: ((in_size, out_size), (next layer))")
	parser.add_argument("--discount_factor", type=float, help="constant to discount future rewards", default=0.99)
	parser.add_argument("--learning_rate", type=float, help="constant to scale parameter updates", default=0.00025)
	parser.add_argument("--optimizer", type=str, help="optimization method for network", 
		choices=('rmsprop', 'graves_rmsprop'), default='graves_rmsprop')
	parser.add_argument("--rmsprop_decay", type=float, help="decay constant for moving average in rmsprop", default=0.95)
	parser.add_argument("--rmsprop_epsilon", type=int, help="constant to stabilize rmsprop", default=0.01)
	# set error_clipping to less than 0 to turn it off
	parser.add_argument("error_clipping", type=str, help="constant at which td-error becomes linear instead of quadratic", default=1.0)
	parser.add_argument("--target_update_frequency", type=int, help="number of steps between target network updates", default=10000)
	parser.add_argument("--memory_capacity", type=int, help="max number of experiences to store in experience memory", default=1000000)
	parser.add_argument("--batch_size", type=int, help="number of transitions sampled from memory during learning", default=32)
	# must set to custom in order to specify custom architecture
	parser.add_argument("--network_architecture", type=str, help="name of prespecified network architecture", 
		choices=("deepmind_nips", "deepmind_nature, custom"), default="deepmind_nature")

	# parser.add_argument("double_dqn", help="use double q-learning algorithm in error target calculation", action=store_true)
	args = parser.parse_args()


	if args.network_architecture == 'deepmind_nature':
		args.conv_kernel_shapes = [
			[8,8,4,32],
			[4,4,32,64],
			[3,3,64,64]]
		args.conv_strides = [
			[1,4,4,1],
			[1,2,2,1],
			[1,1,1,1]]
		args.dense_layer_shapes = [[3136, 512]]
	elif args.network_architecture == 'deepmind_nips':
		args.conv_kernel_shapes = [
			[8,8,4,16],
			[4,4,16,32]]
		args.conv_strides = [
			[1,4,4,1],
			[1,2,2,1]]
		args.dense_layer_shapes = [[2592, 256]]

	if not args.watch:
		record_stats = RecordStats(args)
		training_emulator = AtariEmulator(args, record_stats)
		testing_emulator = AtariEmulator(args, record_stats)
		num_actions = len(training_emulator.get_possible_actions())
		experience_memory ExperienceMemory(args, num_actions)
		q_network = QNetwork(args, num_actions, record_stats)
		agent = DQNAgent(args, q_network, training_emulator, experience_memory, num_actions)
		experiment.run_experiment(args, agent, testing_emulator)

	else:
		training_emulator = AtariEmulator(args, None)
		testing_emulator = AtariEmulator(args, None)
		num_actions = len(training_emulator.get_possible_actions())
		q_network = QNetwork(args, num_actions, None)
		agent = DQNAgent(args, q_network, None, None, num_actions)
		experiment.evaluate_agent(args, agent, testing_emulator)

if __name__ == "__main__":
    main()