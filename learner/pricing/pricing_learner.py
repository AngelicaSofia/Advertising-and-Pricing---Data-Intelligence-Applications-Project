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

class UCB1_Learner(Learner):
    '''UCB1 Learner always selects the arm with the upper confidence bound, calculated with the formula of UCB1. The
    arm is selected as the one that has the maximum value of the average of the samples (calculated as the value
    self.expected_rewards plus a term that is the confidence (computed with the square root of the logarithm of 2 times
    the time divided by the number of times an arm has been pulled).'''
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.expected_rewards = np.zeros(n_arms)
        self.count_pulled_arms = np.zeros(n_arms)
        # Select arm that has the higher upper confidence bound reward
        self.upper_conf_bound = np.zeros(self.n_arms)

    def pull_arm(self):
        """UCB1 samples the first time all the arms, then the arm with the higher upper confidence bound"""
        if self.t < self.n_arms:
            return self.t

        # Compute upper confidence bounds for all arms
        for i in range(10):
            confidence = np.sqrt(2 * np.log(self.t) / (self.count_pulled_arms[i]))
            self.upper_conf_bound[i] = self.expected_rewards[i] + confidence
        # Returns index of arm with higher upper confidence bound
        idxs = np.argmax(self.upper_conf_bound)
        return idxs

    def update(self, pulled_arm, reward):
        """Function that takes the pulled arm and the reward given by it, it updates the reward per arm and the rewards
        of all collected arms and updates the number of times an arm has been pulled"""
        self.t+=1
        self.count_pulled_arms[pulled_arm] += 1
        self.update_observations(pulled_arm, reward)
        # Update array of expected rewards that is simply the average of collected expected rewards for this arm

        # Perform a recursive update fo the average
        self.expected_rewards[pulled_arm] = (self.expected_rewards[pulled_arm] * (self.t - 1) + reward) / self.t

