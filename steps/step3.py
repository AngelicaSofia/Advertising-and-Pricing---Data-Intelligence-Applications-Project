from dia.environments import Scenario
from dia.learner.pricing.pricing_learner import TS_Learner, UCB1_Learner

import numpy as np
import matplotlib.pyplot as plt
import collections
from collections import Counter
import pandas as pd

class Step3:
    """ This step is a pure pricing task. We have a fixed value of number of clicks and cost per click and we want to
    learn the conversion rate curve (talking about aggregate data, so only one conversion rate)."""
    def __init__(self, number_clicks, cost_per_click, lambda_poisson, arms, probabilities, conv_rate):
        self.number_clicks = number_clicks
        self.cost_per_click = cost_per_click
        self.lambda_poisson = lambda_poisson

        self.arms = arms #prices
        self.n_arms = len(self.arms)
        self.probabilities = probabilities #probabilty for every arm
        self.conv_rate = conv_rate

        self.pulled_arms = []

    def objective_function(self, price):
        """This function returns the objective function, having a bid, number of clicks and cost per click fixed. The
        value that is changing is the price"""
        reward = price * self.conv_rate[(price - 1).astype(int)] * self.number_clicks * (1 + self.lambda_poisson) \
                 - (self.cost_per_click * self.number_clicks)
        return reward

    def execute(self, scenario: Scenario, horizon, n_experiment):
        """This function executes the pricing experiment. The time horizon is 360, as we have to consider an horizon of
        1 year and the round is equal to one single day.

        horizon: time horizon of the experiment
        n_experiments: number of experiments to perform
        """
        ts_rewards_per_experiment = []  # store rewards
        ucb1_rewards_per_experiment = []
        #gr_rewards_per_experiment = []  # store rewards
        ucb1_best_arm_per_experiment = []

        for e in range(0, n_experiment):
            env = scenario.pricing_environment
            # simulate interaction
            ts_learner = TS_Learner(n_arms=self.n_arms)
            ucb1_learner = UCB1_Learner(n_arms=self.n_arms)
            #gr_learner = Greedy_Learner(n_arms=n_arms)

            # iterate on number of rounds - simulate interaction between learner and environment
            for t in range(0, horizon):
                # Thompson Sampling TS Learner
                pulled_arm = ts_learner.pull_arm()  # learner compute arm to pull
                reward = env.round(pulled_arm)  # environment compute reward given the pulled arm
                reward *= self.objective_function(self.arms[pulled_arm])
                ts_learner.update(pulled_arm, reward)  # learner updates the rewards
                self.pulled_arms.append(self.arms[pulled_arm])

                # Greedy Learner GR Learner
                #pulled_arm = gr_learner.pull_arm()  # learner compute arm to pull
                #reward = env.round(pulled_arm)  # environment compute reward given the pulled arm
                #gr_learner.update(pulled_arm, reward)  # learner updates the rewards

                # UCB1 Learner
                pulled_arm = ucb1_learner.pull_arm()  # learner compute arm to pull
                reward = env.round(pulled_arm)  # environment compute reward given the pulled arm
                reward *= self.objective_function(self.arms[pulled_arm])
                ucb1_learner.update(pulled_arm, reward)  # learner updates the rewards

            # store value of collected rewards
            ts_rewards_per_experiment.append(np.sum(ts_learner.collected_rewards))
            ucb1_rewards_per_experiment.append(np.sum(ucb1_learner.collected_rewards))
            ucb1_best_arm_per_experiment.append(np.argmin(2 * np.log(horizon) / ucb1_learner.count_pulled_arms))
            #gr_rewards_per_experiment.append(np.sum(gr_learner.collected_rewards))

        # select best arm TS
        self.best_arm = self.select_best_arm()
        # select best arm UCB1
        #self.best_arm_ucb1 = np.argmin(2 * np.log(horizon) / ucb1_learner.count_pulled_arms)
        self.best_arm_ucb1 = np.max(ucb1_best_arm_per_experiment) - 1.0

        print("The best arm (price TS) is: " + str(self.best_arm))
        print("The best arm (price UCB1) is: " + str(self.best_arm_ucb1))
        data = np.array(self.pulled_arms)
        labels = []
        for i in self.arms:
            count = np.count_nonzero(data==i)
            labels.append(count)

        fig, ax = plt.subplots(1, 1)
        #ax.hist(self.pulled_arms, histtype='bar', ec='blue')
        plt.bar(self.arms, height=labels)
        ax.set_title('Histogram of arms pulled')
        ax.set_xlabel('Arms')
        ax.set_ylabel('Number of times each arm was pulled')
        ax.set_xticks(np.arange(self.n_arms)+1)
        ax.set_xticklabels(self.arms)
        rects = ax.patches
        best = self.best_arm.astype(int)

        for rect, label in zip(rects, labels):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, height + 5, label,
                    ha='center', va='bottom')
        plt.plot()
        plt.ylim(0, labels[best-1]+300)
        plt.show()

    def select_best_arm(self):
        return max(set(self.pulled_arms), key=self.pulled_arms.count)