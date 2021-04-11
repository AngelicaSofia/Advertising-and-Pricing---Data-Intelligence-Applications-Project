# Learners

import numpy as np
import math


class Learner:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.t = 0
        # rewards_per_arm is a list of list
        # External list is the #arms, length of internal list is the number of times we pull a given arm
        self.rewards_per_arm = [[] for i in range(n_arms)]
        self.collected_rewards = np.array([])

    def pull_arm(self):
        pass

    def update_observations(self, pulled_arm, reward):
        """Function that updates observations once the reward is returned by the environment"""
        self.rewards_per_arm[pulled_arm].append(reward) # update list of the arm pulled by the learner
        self.collected_rewards = np.append(self.collected_rewards, reward)


class TS_Learner(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        # beta distribution is characterized by 2 parameters (alpha and beta)
        self.beta_parameters = np.ones((n_arms, 2))

    def pull_arm(self):
        """TS samples values from a beta distribution, then selects the arm with the higher value"""
        idx = np.argmax(np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1]))
        # update at each round t the values of the beta distribution
        return idx

    def update(self, pulled_arm, reward):
        '''Function that updates the beta parameters given the reward'''
        self.t += 1 # update round value
        self.update_observations(pulled_arm, reward) # update observations

        success = 0
        # If reward is positive, then we have a success, on the contrary, we have a failure
        if reward > 0:
            success = 1
        elif reward <= 0:
            success = 0

        # First parameter: How many successes we have
        self.beta_parameters[pulled_arm, 0] = self.beta_parameters[pulled_arm, 0] + success
        # Second parameter: Opposite
        self.beta_parameters[pulled_arm, 1] = self.beta_parameters[pulled_arm, 1] + 1.0 - success

"""
class Greedy_Learner(Learner):
    '''Greedy Learner that has to compute at each round, the value of the expected reward for the pulled arm'''
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.expected_rewards = np.zeros(n_arms)

    def pull_arm(self):
        Greedy Learner selects the arm to pull at each round t, by maximizing th expected reward array.

        Note that we assume that each arm is pulled at least once, so in the first rounds all arms are pulled (arm 0 at
        round 0, arm 1 at round 1, till all arms are pulled once).
        
        if self.t < self.n_arms:
            return self.t
        # Select arm that maximizes the expected reward
        idxs = np.argwhere(self.expected_rewards == self.expected_rewards.max()).reshape(-1)
        # As we could have multiple indexes, we randomly select one of the index that maximizes the expected reward
        chosen_pulled_arm = np.random.choice(idxs)
        return chosen_pulled_arm

    def update(self, pulled_arm, reward):
        Function that takes the pulled arm and the reward given by it
        self.t+=1
        self.update_observations(pulled_arm, reward)
        # Update array of expected rewards that is simply the average of collected expected rewards for this arm
        # Perform a recursive update fo the average
        self.expected_rewards[pulled_arm] = (self.expected_rewards[pulled_arm] * (self.t - 1) + reward) / self.t

"""