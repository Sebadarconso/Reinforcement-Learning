import os, sys, numpy
module_path = os.path.abspath(os.path.join('/Users/sebastianodarconso/Desktop/magistrale_lab/reinforcement_learning/Lab3/RL-Lab-main/tools'))
if module_path not in sys.path: sys.path.append(module_path)
from DangerousGridWorld import GridWorld


def on_policy_mc( environment, maxiters=5000, eps=0.3, gamma=0.99 ):
	"""
	Performs the on policy version of the every-visit MC control
	
	Args:
		environment: OpenAI Gym environment
		maxiters: timeout for the iterations
		eps: random value for the eps-greedy policy (probability of random action)
		gamma: gamma value, the discount factor for the Bellman equation
		
	Returns:
		policy: 1-d dimensional array of action identifiers where index `i` corresponds to state id `i`
	"""

	p = [[0 for _ in range(environment.action_space)] for _ in range(environment.observation_space)]   
	Q = [[0 for _ in range(environment.action_space)] for _ in range(environment.observation_space)]
	returns = [[[]for _ in range(environment.action_space)] for _ in range(environment.observation_space)]

	for episode in range(maxiters):
		e = environment.sample_episode(p)
		g = 0 
		for state, action, reward in reversed(e):
			g = gamma * g + reward
			returns[state][action].append(g)
			Q[state][action] = numpy.sum(returns[state][action]) / len(returns[state][action])
			a_star = numpy.argmax(Q[state])

			for act in range(environment.action_space):
				if act == a_star:
					p[state][act] = 1 - eps + (eps / environment.action_space)
				else:
					p[state][act] = eps / environment.action_space

			
	deterministic_policy = [numpy.argmax(p[state]) for state in range(environment.observation_space)]	
	return deterministic_policy


def main():
	print( "\n*************************************************" )
	print( "*  Welcome to the third lesson of the RL-Lab!   *" )
	print( "*           (Monte Carlo Methods)               *" )
	print( "**************************************************" )

	print("\nEnvironment Render:")
	env = GridWorld()
	env.render()

	print( "\n3) MC On-Policy" )
	mc_policy = on_policy_mc( env )
	env.render_policy( mc_policy )
	print( "\tExpected reward following this policy:", env.evaluate_policy(mc_policy) )
	

if __name__ == "__main__":
	main()
