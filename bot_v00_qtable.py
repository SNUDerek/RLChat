import numpy as np
from environment import Environment0

# slot list
slot_list = ['greeting', 'product', 'query', 'answer', 'anything_else', 'goodbye']

# create new environment
env = Environment0(slot_list)

print("table lookup Q-learning\n")

# initialize table with all zeros
# Q = np.zeros([env.state_len, env.action_len])
# todo: intialize with randomized (small) values?
Q = np.multiply(np.random.rand(env.state_len, env.action_len), 0.01)

# set learning parameters (learning rate lr, discount factor y [should be ~0.9])
lr = .60
y = .9
num_episodes = 250000

#create lists to contain total rewards and steps per episode
doneList = []
rList = []
tmp_rList = []
tmp_idx = 0

for i in range(num_episodes):

    # reset environment and get first new observation
    s = env.resetenv()
    reward_all = 0
    d = False
    j = 0

    # Q-Table learning algorithm
    while j < 20:
        j += 1
        tmp_idx += 1

        # choose an action by greedily (with noise) picking from Q table
        # todo: disabled reducing randomness over time for more exploration
        a = np.argmax(np.multiply(Q[s,:] + np.random.randn(1, env.action_len), 0.5)) # * (1./(i+1)))

        # get new state and reward from environment
        s1, r, d = env.step(a)

        # update Q-Table with new knowledge
        # according to Bellman eqn
        Q[s, a] = Q[s, a] + lr*(r + y*np.max(Q[s1, :]) - Q[s, a])
        reward_all += r
        s = s1

        # end convo?
        if d == True:

            if r > 0:
                doneList.append(1)
            else:
                doneList.append(0)

            break

    #jList.append(j)
    rList.append(reward_all)

    # for increment count
    tmp_rList.append(reward_all)

    if i % 5000 == 0:
        #print("Score over time: " +  str(sum(rList)/num_episodes), i)
        print("Score this increment: " + str(sum(tmp_rList) / tmp_idx), i)
        tmp_rList = []
        tmp_idx = 0

print("number of succesful episodes: " + str(sum(doneList)) + "%")
print("Final Q-Table Values")

print(slot_list)

for si, row in enumerate(Q.tolist()):
    print(si, env.states[si], row)