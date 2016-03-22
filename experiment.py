

def evaluate_agent(args, agent, test_emulator):
	step = 0
	games = 0
	reward = 0.0
	test_emulator.reset()
	screen = test_emulator.preprocess()

	while (step < args.testing_steps) and (games < args.testing_games):
		while not test_emulator.isGameOver():
			action = agent.test_step(screen)
			results = test_emulator.run_step(action)
			screen = results[0]
			reward += results[2]
			step +=1

		games += 1
		agent.test_state = test_emulator.reset()

	return [reward / games, games]



def run_experiment(args, agent, test_emulator, test_stats):
	
	agent.run_random_exploration()

	for epoch in range(1, args.epochs + 1):

		if epoch == 1:
			agent.run_epoch(args.epoch_length - agent.random_exploration_length, epoch)
		else:
			agent.run_epoch(args.epoch_length, epoch)

		results = evaluate_agent(args, agent, test_emulator)
		print("Score for epoch {0}: {1}".format(epoch, results[0]))
		test_stats.record(epoch)