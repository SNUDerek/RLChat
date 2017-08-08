import numpy as np
import itertools

# Q-table environment
# this assumes each 'action' speech-act is just a request for the slot info
# so slots = actions

class Environment0(object):
    '''
    environment class for Q-table learning. 
    the step() function advances the conversation
    the reset() function resets the environment
    
    Attributes:
    slot_list: list of possible slots
    noise_level: level of uncertainty to affect output (as % of time)
    annoyance_level: max level of bad lines before termination
    i2x: dict of indices to slot name strings
    x2i: dict of slot name strings to indices
    states: list of binary (filled/unfilled) state vectors
    state_idx: dict of string-converted state vectors to indices (for Q table)
    annoyance: current customer annoyance level
    current_state: current conversation state
    customer_state: customer's opinion of current state
    turnnumber: turns, for rule-based counting
    '''

    # todo: add memory, functionality for slot ~values~

    def __init__(self, slot_list, noise_level=0.05, annoyance_level=7):

        self.slot_list = slot_list
        self.noise_level = noise_level
        self.annoyance = 0
        self.annoyance_level = annoyance_level

        # init state info
        self.i2x, self.x2i, self.states, self.state_idx = self.init_state_info(slot_list)

        # init to empty state
        self.current_state = np.copy(self.states[0])
        self.customer_state = self.current_state
        self.turnnumber = 0

        self.state_len = len(self.states)
        # todo: change this if adding better action representation
        self.action_len = len(self.slot_list)

    def init_state_info(self, slot_list):
        '''turn slot list into state representation'''

        # dictionaries to convert between string slot values and indices
        i2x = dict(enumerate(slot_list)) # index to string mapping
        x2i = {v:k for k, v in list(enumerate(slot_list))} # string to index mapping

        # represent states as binary vectors of len(states)
        # where 0 = no_value, 1 = value
        # see: https://stackoverflow.com/questions/14931769/how-to-get-all-combination-of-n-binary-value
        # looks like [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0],...]
        _states = [list(i) for i in itertools.product([0, 1], repeat=len(slot_list))]

        # dictionary to get state index (for ease of use) - TURN LIST TO STRING!!!!
        _state_idx = {str(v):k for k, v in list(enumerate(_states))}

        return i2x, x2i, _states, _state_idx


    def step(self, action):
        '''advance the conversation one step'''

        # update customer state info
        next_state, reward, hang_up, action_idx = self._customer(action)

        # add noisy output
        self.current_state = self._addnoise(next_state, action)

        self.turnnumber += 1

        return self.state_idx[str(self.current_state)], reward, hang_up


    def resetenv(self):
        '''reset the simulation'''
        self.current_state = np.copy(self.states[0]).tolist()
        self.customer_state = self.current_state
        self.turnnumber = 0
        self.annoyance = 0
        return self.state_idx[str(self.current_state)]


    def _customer(self, action):
        '''rule-based approach based on ten example conversations
        this just returns action for now (perfect response),
        but calculates 'annoyance' based on 'bad' orders
        e.g. saying 'goodbye' before 'action' is filled, etc'''
        # todo: NB: THIS DEPENDS ON FIXED STATE REPRESENTATION!!!

        hang_up = False # end convo
        reward = 0

        last_state = np.copy(self.customer_state).tolist()
        # update customer state with action query
        # (= customer knows it's been asked)
        self.customer_state[action] = 1

        # turn values to strings for human-readable editing of rules
        action = self.i2x[action] # action to string


        # IMMEDIATE CONVERSATION-KILLERS:

        # ...saying goodbye or anything_else before giving answer
        if action == 'goodbye' or action == 'anything_else' and last_state[self.x2i['answer']] == 0:

            self.annoyance += self.annoyance_level
            hang_up = True
            reward = -1

            # return the state, reward, early termination (death), and the idx (for noise)
            return self.customer_state, reward, hang_up, self.x2i[action]

        # # ...giving answer before finding out query
        # if action == 'answer' and (self.customer_state[self.x2i['query']]):
        #
        #     self.annoyance += self.annoyance_level
        #     hang_up = True

        # ...giving answer before finding out query and product
        if action == 'answer' and (last_state[self.x2i['query']] == 0 \
                                    and last_state[self.x2i['product']] == 0):

            self.annoyance += self.annoyance_level
            hang_up = True
            reward = -1

            # return the state, reward, early termination (death), and the idx (for noise)
            return self.customer_state, reward, hang_up, self.x2i[action]

        # MINOR ANNOYANCES

        # if no greeting in the first few turns, penalize (minor)
        if action != 'greeting' and self.turnnumber < 2 and last_state[self.x2i['greeting']] == 0:

            self.annoyance += 1

        # if give greeting after the first few turns, penalize (moderate)
        if action == 'greeting' and self.turnnumber > 2:

            self.annoyance += 6
            reward = -0.05

        # if goodbye without 'anything else', penalize (minor)
        if action != 'goodbye' and last_state[self.x2i['anything_else']] == 0:

            self.annoyance += 1

        # if asking about ANY slot that's already filled, penalize (moderate)
        # i.e. asking repeat questions
        if last_state[self.x2i[action]] == 1:

            self.annoyance += 5


        # MINOR PLEASANTRIES
        # if do greeting in the first few turns, add small reward
        if action == 'greeting' and self.turnnumber < 3:

            reward = 0.1

        # if confirmed both product and query, add small reward
        # maybe bc customer says thank you or something
        if action == 'answer' and (self.customer_state[self.x2i['answer']] == 0 \
                                   and self.customer_state[self.x2i['query']] == 1 \
                                   and self.customer_state[self.x2i['product']] == 1):

            reward = 0.1


        # early termination if too annoyed
        if self.annoyance > self.annoyance_level:

            hang_up = True
            reward = -1

            # return the state, reward, early termination (death), and the idx (for noise)
            return self.customer_state, reward, hang_up, self.x2i[action]

        # else, if end goal met (asked problem and query, gave answer, said goodbye), reward best
        if action == 'goodbye' and (last_state[self.x2i['query']] == 1 \
                                   and last_state[self.x2i['product']] == 1 \
                                   and last_state[self.x2i['answer']] == 1):

            hang_up = True
            reward = 1000

            # return the state, reward, early termination (death), and the idx (for noise)
            return self.customer_state, reward, hang_up, self.x2i[action]


        # else, if end goal met (asked problem, gave answer, said goodbye), reward small
        elif action == 'goodbye' and (last_state[self.x2i['query']] == 1 \
                                   and last_state[self.x2i['answer']] == 1):

            hang_up = True
            reward = 100

            # return the state, reward, early termination (death), and the idx (for noise)
            return self.customer_state, reward, hang_up, self.x2i[action]


        # return the state, reward, early termination (death), and the idx (for noise)
        return self.customer_state, reward, hang_up, self.x2i[action]


    def _addnoise(self, state, action):

        flip_idx = action

        # choose action to take beside action
        # todo: also not goodbye?
        while self.i2x[flip_idx] not in ['goodbye', self.i2x[action]]:
            flip_idx = np.random.randint(0, len(self.slot_list))

        chance = np.random.random()

        # at rate of noise_level %, flip last state and flip another
        if chance <= self.noise_level:

            state[action] = 0 # undo true value
            # randomly switch another value
            if state[flip_idx] == 0:
                state[flip_idx] = 1
            else:
                state[flip_idx] = 1

        return state



