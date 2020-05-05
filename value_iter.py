##################################
# Create env
import gym

env = gym.make('FrozenLake-v0')
env = env.env
print(env.__doc__)
print("")

#################################
# Some basic imports and setup
# Let's look at what a random episode looks like.

import numpy as np, numpy.random as nr, gym
import matplotlib.pyplot as plt

# %matplotlib inline
np.set_printoptions(precision=3)

# Seed RNGs so you get the same printouts as me
env.seed(0)
# from gym.spaces import prng prng.seed(10)
env.action_space.np_random.seed(10)
# Generate the episode
env.reset()


# for t in range(100):
#     env.render()
#     a = env.action_space.sample()
#     ob, rew, done, _ = env.step(a)
#     if done:
#         break
# assert done
# env.render();

#################################
# Create MDP for our env
# We extract the relevant information from the gym Env into the MDP class below.
# The `env` object won't be used any further, we'll just use the `mdp` object.

class MDP(object):
    def __init__(self, P, nS, nA, desc=None):
        self.P = P  # state transition and reward probabilities, explained below
        self.nS = nS  # number of states
        self.nA = nA  # number of actions
        self.desc = desc  # 2D array specifying what each grid cell means (used for plotting)


mdp = MDP({s: {a: [tup[:3] for tup in tups] for (a, tups) in a2d.items()} for (s, a2d) in env.P.items()}, env.nS,
          env.nA, env.desc)


# print("")
# print("mdp.P is a two-level dict where the first key is the state and the second key is the action.")
# print("The 2D grid cells are associated with indices [0, 1, 2, ..., 15] from left to right and top to down, as in")
# print(np.arange(16).reshape(4,4))
# print("Action indices [0, 1, 2, 3] correspond to West, South, East and North.")
# print("mdp.P[state][action] is a list of tuples (probability, nextstate, reward).\n")
# print("For example, state 0 is the initial state, and the transition information for s=0, a=0 is \nP[0][0] =", mdp.P[0][0], "\n")
# print("As another example, state 5 corresponds to a hole in the ice, in which all actions lead to the same state with probability 1 and reward 0.")
# for i in range(4):
#     print("P[5][%i] =" % i, mdp.P[5][i])
# print("")

#################################
# Programing Question No. 1 - implement where required.

def value_iteration(mdp, gamma, nIt):
    """
    Inputs:
        mdp: MDP
        gamma: discount factor
        nIt: number of iterations, corresponding to n above
    Outputs:
        (value_functions, policies)

    len(value_functions) == nIt+1 and len(policies) == nIt
    """
    print("Iteration | max|V-Vprev| | # chg actions | V[0]")
    print("----------+--------------+---------------+---------")
    Vs = [np.zeros(mdp.nS)]  # list of value functions contains the initial value function V^{(0)}, which is zero
    pis = []
    # rearranging P and R for convenient calculation.
    P = [[[0 for s in range(mdp.nS)] for a in range(mdp.nA)] for s_tag in range(mdp.nS)]
    R = [[[0 for s in range(mdp.nS)] for a in range(mdp.nA)] for s_tag in range(mdp.nS)]
    for s in range(mdp.nS):
        for a in range(mdp.nA):
            tuples_list = mdp.P[s][a]
            for tup in tuples_list:
                P[s][a][tup[1]] += tup[0]
                R[s][a][tup[1]] += tup[2]

    for it in range(nIt):
        oldpi = pis[-1] if len(pis) > 0 else None  # \pi^{(it)} = Greedy[V^{(it-1)}]. Just used for printout
        Vprev = Vs[-1]  # V^{(it)}

        # Your code should fill in meaningful values for the following two variables
        # pi: greedy policy for Vprev (not V),
        #     corresponding to the math above: \pi^{(it)} = Greedy[V^{(it)}]
        #     ** it needs to be numpy array of ints **
        # V: bellman backup on Vprev
        #     corresponding to the math above: V^{(it+1)} = T[V^{(it)}]
        #     ** numpy array of floats **

        V = np.zeros(mdp.nS)
        pi = np.zeros(mdp.nS, dtype=int)
        for s in range(mdp.nS):
            a_V = np.zeros(mdp.nA)
            for a in range(mdp.nA):
                a_V[a] = sum([P[s][a][s_tag] * (R[s][a][s_tag] + gamma * Vprev[s_tag]) for s_tag in range(mdp.nS)])
            pi[s] = np.argmax(a_V)
            V[s] = a_V[pi[s]]

        # V = Vprev # REPLACE THIS LINE WITH YOUR CODE
        # pi = oldpi # REPLACE THIS LINE WITH YOUR CODE

        max_diff = np.abs(V - Vprev).max()
        nChgActions = "N/A" if oldpi is None else (pi != oldpi).sum()
        print("%4i      | %6.5f      | %4s          | %5.3f" % (it, max_diff, nChgActions, V[0]))
        Vs.append(V)
        pis.append(pi)
    return Vs, pis


GAMMA = 0.95  # we'll be using this same value in subsequent problems
Vs_VI, pis_VI = value_iteration(mdp, gamma=GAMMA, nIt=100)

#################################
# Below is code for illustrating the progress of value iteration.
# Your optimal actions are shown by arrows.
# At the bottom, the value of the different states are plotted.

plt.figure()
for s in range(mdp.nS):
    v_s = [v_i[s] for v_i in Vs_VI]
    plt.plot(v_s, label=str(s))
plt.legend()
plt.title('state value to iteration')
# plt.savefig('plots/q1/state value to iteration')

i = 0
for (V, pi) in zip(Vs_VI[:100], pis_VI[:100]):
    i += 1
    plt.figure(figsize=(3, 3))
    plt.imshow(V.reshape(4, 4), cmap='gray', interpolation='none', clim=(0, 1))
    ax = plt.gca()
    ax.set_xticks(np.arange(4) - .5)
    ax.set_yticks(np.arange(4) - .5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    Y, X = np.mgrid[0:4, 0:4]
    a2uv = {0: (-1, 0), 1: (0, -1), 2: (1, 0), 3: (0, 1)}
    Pi = pi.reshape(4, 4)
    for y in range(4):
        for x in range(4):
            a = Pi[y, x]
            u, v_s = a2uv[a]
            plt.arrow(x, y, u * .3, -v_s * .3, color='m', head_width=0.1, head_length=0.1)
            plt.text(x, y, str(env.desc[y, x].item().decode()),
                     color='g', size=12, verticalalignment='center',
                     horizontalalignment='center', fontweight='bold')
    plt.grid(color='b', lw=2, ls='-')
    plt.title('policy{}'.format(i))
    # plt.savefig('plots/q1/policy{}.png'.format(i))
