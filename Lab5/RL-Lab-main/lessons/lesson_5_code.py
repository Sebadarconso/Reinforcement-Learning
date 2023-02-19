import os, sys, numpy, random
module_path = os.path.abspath(os.path.join('/Users/sebastianodarconso/Desktop/magistrale_lab/reinforcement_learning/Lab5/RL-Lab-main/tools'))
if module_path not in sys.path: sys.path.append(module_path)
from DangerousGridWorld import GridWorld


def epsilon_greedy(q, state, epsilon):
	"""
	Epsilon-greedy action selection function
	
	Args:
		q: q table
		state: agent's current state
		epsilon: epsilon parameter
	
	Returns:
		action id
	"""
	if numpy.random.random() < epsilon:
		return numpy.random.choice(q.shape[1])
	return q[state].argmax()


def dynaQ( environment, maxiters=250, n=10, eps=0.3, alfa=0.3, gamma=0.99 ):

	Q = numpy.zeros((environment.observation_space, environment.action_space))
	M = numpy.array([[[None, None] for _ in range(environment.action_space)] for _ in range(environment.observation_space)])

	observed = list([])

	for _ in range(maxiters):

		s = environment.random_initial_state()
		a = epsilon_greedy(Q, s, eps)

		new_state = environment.sample(a, s)
		reward = environment.R[new_state]
		visited = list([s, a])
		if visited not in observed:
			observed.append(visited)

		val = [0 for _ in environment.actions]
		for action in environment.actions:
			val[action] = Q[new_state, action]
		
		Q[s, a] += alfa * (reward + gamma * max(val) - Q[s, a])
		M[s, a] = [reward, new_state]

		for _ in range(n):

			index = random.choice(range(len(observed)))
			s, a = observed[index]

			reward, new_state_f = M[s, a]
			val = [0 for _ in environment.actions]
			for action in environment.actions:
				val[action] = Q[new_state_f, action]
			
			Q[s, a] += alfa * (reward + gamma * max(val) - Q[s, a])

	policy = Q.argmax(axis=1) 
	return policy



def main():
	print( "\n************************************************" )
	print( "*   Welcome to the fifth lesson of the RL-Lab!   *" )
	print( "*                  (Dyna-Q)                      *" )
	print( "**************************************************" )

	print("\nEnvironment Render:")
	env = GridWorld( deterministic=True )
	env.render()

	print( "\n6) Dyna-Q" )
	dq_policy_n00 = dynaQ( env, n=0  )
	dq_policy_n25 = dynaQ( env, n=25 )
	dq_policy_n50 = dynaQ( env, n=50 )

	env.render_policy( dq_policy_n50 )
	print()
	print( f"\tExpected reward with n=0 :", env.evaluate_policy(dq_policy_n00) )
	print( f"\tExpected reward with n=25:", env.evaluate_policy(dq_policy_n25) )
	print( f"\tExpected reward with n=50:", env.evaluate_policy(dq_policy_n50) )
	
	

if __name__ == "__main__":
	main()