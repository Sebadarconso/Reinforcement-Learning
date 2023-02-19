import os, sys
import numpy as np
module_path = os.path.abspath(os.path.join('/Users/sebastianodarconso/Desktop/magistrale_lab/reinforcement_learning/Lab2/RL-Lab-main/tools'))
if module_path not in sys.path: sys.path.append(module_path)
from DangerousGridWorld import GridWorld


def value_iteration(environment, maxiters=300, discount=0.9, max_error=1e-3):
	"""
	Performs the value iteration algorithm for a specific environment
	
	Args:
		environment: OpenAI Gym environment
		maxiters: timeout for the iterations
		discount: gamma value, the discount factor for the Bellman equation
		max_error: the maximum error allowd in the utility of any state
		
	Returns:
		policy: 1-d dimensional array of action identifiers where index `i` corresponds to state id `i`
	"""
	
	U_1 = [0 for _ in range(environment.observation_space)] # vector of utilities for states S
	delta = 0 # maximum change in the utility o any state in an iteration
	U = U_1.copy()
	#
	# YOUR CODE HERE!
	#

	while True:
		U = U_1.copy()
		delta = 0
		for state in range(environment.observation_space):
			sum_actions = [0 for _ in range(environment.action_space)]
			for action in range(environment.action_space):
				for next_state in range(environment.action_space):
					sum_actions[action] += environment.transition_prob(state, action, next_state) * U[next_state]
			
			if state == environment.goal_state or state == environment.death:
			 	U_1[state] = environment.R[state]
			else: 
				U_1[state] = environment.R[state] + discount * max(sum_actions)
		
			if abs(U_1[state] - U[state]) > discount:
				delta = abs(U_1[state] - U[state])
		
		if delta < max_error * (1 - discount) / discount:
			break
	
	return environment.values_to_policy( U )

	

def policy_iteration(environment, maxiters=300, discount=0.9, maxviter=10):
	"""
	Performs the policy iteration algorithm for a specific environment
	
	Args:
		environment: OpenAI Gym environment
		maxiters: timeout for the iterations
		discount: gamma value, the discount factor for the Bellman equation
		maxviter: number of epsiodes for the policy evaluation
		
	Returns:
		policy: 1-d dimensional array of action identifiers where index `i` corresponds to state id `i`
	"""
	
	p = [0 for _ in range(environment.observation_space)] #initial policy    
	U = [0 for _ in range(environment.observation_space)] #utility array
	
	# 1) Policy Evaluation
	#
	# YOUR CODE HERE!
	#
	i = 0
	while True:
		U_i = [0 for _ in range(environment.observation_space)]
		for _ in range(maxiters):
			i += 1
			for _ in range(maxviter):
				val = [0 for _ in range(environment.observation_space)]
				for state in range(environment.observation_space):
					for next_state in range(environment.observation_space):
						val[state] += environment.transition_prob(state, p[state], next_state) * U_i[next_state]
					
					if state == environment.goal_state or state == environment.death:
						U_i[state] = environment.R[state]
					else: 
						U_i[state] = environment.R[state] + (discount * val[state])

		U = U_i.copy()
		unchanged = True
		
		for state in range(environment.observation_space):
			val_state = [0 for _ in range(environment.observation_space)]
			val_action = [0 for _ in range(environment.action_space)]

			for action in range(environment.action_space):
				for next_state in range(environment.observation_space):
					val_action[action] += environment.transition_prob(state, action, next_state) * U[next_state]
				
				for next_state in range(environment.observation_space):
					val_state[state] += environment.transition_prob(state, p[state], next_state) * U[next_state]

				if max(val_action) > val_state[state]:
					p[state] = np.argmax(val_action)
					unchanged = False
			
		
		if unchanged:
			break
	# 2) Policy Improvement
	#
	# YOUR CODE HERE!
	#    
	
	return p



def main():
	print( "\n************************************************" )
	print( "*  Welcome to the second lesson of the RL-Lab! *" )
	print( "*    (Policy Iteration and Value Iteration)    *" )
	print( "************************************************" )

	print("\nEnvironment Render:")
	env = GridWorld()

	print( "\n1) Value Iteration:" )
	vi_policy = value_iteration( env )
	env.render_policy( vi_policy )
	print( "\tExpected reward following this policy:", env.evaluate_policy(vi_policy) )

	print( "\n1) Policy Iteration:" )
	pi_policy = policy_iteration( env )
	env.render_policy( pi_policy )
	print( "\tExpected reward following this policy:", env.evaluate_policy(pi_policy) )


if __name__ == "__main__":
	main()