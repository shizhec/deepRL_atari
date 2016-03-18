

def evaluate_agent(agent, test_emulator, testing_steps, testing_games):
	step = 0
	games = 0
	reward = 0.0
	test_emulator.reset()
	screen = test_emulator.preprocess()

	while (step < testing_steps) and (games < testing_games):
		while not test_emulator.isGameOver():
			action = agent.test_step(screen)
			results = test_emulator.run_step(action)
			screen = results[0]
			reward += results[2]
			step +=1

		games += 1
		test_emulator.reset()
		agent.test_reset()

	return [reward / games, games]



def run_experiment(agent, num_epochs, epoch_length, test_emulator, testing_steps, testing_games):
	
	agent.run_random_exploration()

	for epoch in range(num_epochs):

		if epoch == 0:
			agent.run_epoch(epoch_length - agent.random_exploration_length)
		else:
			agent.run_epoch(epoch_length)

		results = evaluate_agent(agent, test_emulator, testing_steps, testing_games)

		print("Score for epoch {0}: {1}".format(epoch, results[0]))  # TODO: store results in csv