from dia.environments import Scenario
from dia.environments import pricing_environment
from dia.learner.pricing.pricing_learner import TS_Learner
from dia.learner.advertising.gp_learner import GP_Learner
from dia.environments import function_manager as fm
from dia.utils import logger

import numpy as np
import matplotlib.pyplot as plt
import collections
from collections import Counter
import pandas as pd
import random

class Step5:
    """ This step is a pure ... TODO"""
    def __init__(self, lambda_poisson, arms, probabilities, conv_rate, n_obs, noise_std_n_clicks,
                 noise_std_cost_x_click, updated_p):
        self.lambda_poisson = lambda_poisson

        self.arms = arms #bids
        self.n_arms = len(self.arms)
        self.successes = probabilities #probabilty for every arm
        self.init_probabilities = probabilities
        self.updated_probabilities = updated_p
        self.conv_rate = conv_rate
        self.n_obs = n_obs
        self.noise_std_n_clicks = noise_std_n_clicks
        self.noise_std_cost_x_click = noise_std_cost_x_click
        self.estimation_round = 1
        self.exp = 0 # experiment
        # estimation of functions is not done every round, but every ten rounds.
        self.temp_estimated_cost = [0] * 10
        self.temp_estimated_n_clicks = [0] * 10

        self.pulled_arms = []
        # Lists to verify that the reward for the arm is positive
        self.number_pulls_for_arm = [0] * len(self.arms)
        self.number_positive_rewards_for_arm = [0] * len(self.arms)

        self.sigma_reward = 0
        self.sigma_cost_per_click = np.zeros(len(self.arms))
        self.sigma_n_clicks = np.zeros(len(self.arms))

        self.lambda_values_poisson_months = np.zeros(360) # horizon one year

    def generate_observations(self, f_to_estimate, x):
        """Function to generate observations. Considering number of clicks and cost per click, the function depends on
        the bid (x == bid)."""
        if f_to_estimate == "n_clicks":
            # x = bid
            n = 0
            for customer_class in range(1, 4):
                n = n + fm.get_n_click(x, customer_class)
            # Division in classes is not known, so an average of the curve is considered
            n = n / 3
            return n + np.random.normal(0, self.noise_std_n_clicks, size=n.shape)
        if f_to_estimate == "cost_per_click":
            # x = bid
            return fm.get_cost_x_click(x) + np.random.normal(0, self.noise_std_cost_x_click,
                                                             size=fm.get_cost_x_click(x).shape)
        return

    def estimate_n_clicks(self, bid):
        gp_learner = GP_Learner(self.noise_std_n_clicks)
        x_obs = np.array([])
        y_obs = np.array([])
        for i in range(0, self.n_obs):
            gp_learner = GP_Learner(self.noise_std_n_clicks)

            new_x_obs = np.random.choice(self.arms, 1)
            # 1 is the customer class
            new_y_obs = self.generate_observations("n_clicks", new_x_obs)

            x_obs = np.append(x_obs, new_x_obs.astype(float))
            y_obs = np.append(y_obs, new_y_obs.astype(float))

            X = np.atleast_2d(x_obs).T
            Y = y_obs.ravel()

            gp_learner.learn(X, Y, self.arms)
            """
            fig = plt.figure(i + self.n_obs*(self.estimation_round)/10)
            plt.plot(gp_learner.x_pred, ((fm.get_n_click(gp_learner.x_pred, 1) + fm.get_n_click(gp_learner.x_pred, 2) +
                                          fm.get_n_click(gp_learner.x_pred, 3))/3), 'r', label=r'$n clicks(bid)$')
            plt.plot(X.ravel(), Y, 'ro', label=u'observed clicks')
            plt.plot(gp_learner.x_pred, gp_learner.y_pred, 'b-', label=u'predicted clicks')
            plt.fill(np.concatenate([gp_learner.x_pred, gp_learner.x_pred[::-1]]),
                     np.concatenate([gp_learner.y_pred - 1.96 * gp_learner.sigma,
                                     (gp_learner.y_pred + 1.96 * gp_learner.sigma)[::-1]]),
                     alpha=.5, fc='b', ec='None', label='95% confidence interval')
            plt.xlabel('$bid$')
            plt.ylabel('$n_clicks(bid)$')
            plt.legend(loc='lower right')
            #plt.show()
            if i == self.n_obs - 1:
                plt.savefig("plots/bidding and learning (step 5)/n_clicks_experiment_{}_estimation_month_{}.png".
                            format(self.exp, int(self.estimation_round/30)))
            plt.close(fig)
            """

        bid_value = np.array([bid])
        predicted_n_clicks = gp_learner.gp.predict(bid_value.reshape(1, -1))
        # predict number of clicks with indicated bid
        float(predicted_n_clicks)
        for i in self.arms:
            b = np.array([i]) # bid value
            pred_n_c = gp_learner.gp.predict(b.reshape(1, -1))
            self.temp_estimated_n_clicks[i-1] = pred_n_c
        self.sigma_n_clicks = gp_learner.sigma
        return predicted_n_clicks

    def estimate_cost_x_click(self, bid):
        gp_learner = GP_Learner(self.noise_std_n_clicks)
        x_obs = np.array([])
        y_obs = np.array([])
        for i in range(0, self.n_obs):
            gp_learner = GP_Learner(self.noise_std_cost_x_click)

            new_x_obs = np.random.choice(self.arms, 1)
            new_y_obs = self.generate_observations("cost_per_click", new_x_obs)

            x_obs = np.append(x_obs, new_x_obs.astype(float))
            y_obs = np.append(y_obs, new_y_obs.astype(float))

            X = np.atleast_2d(x_obs).T
            Y = y_obs.ravel()

            gp_learner.learn(X, Y, self.arms)
            """
            fig = plt.figure(i + 1000 + self.n_obs*(self.estimation_round)/10)
            plt.plot(gp_learner.x_pred, fm.get_cost_x_click(gp_learner.x_pred), 'r', label=r'$cost(bid)$')
            plt.plot(X.ravel(), Y, 'ro', label=u'observed cost')
            plt.plot(gp_learner.x_pred, gp_learner.y_pred, 'b-', label=u'predicted cost')
            plt.fill(np.concatenate([gp_learner.x_pred, gp_learner.x_pred[::-1]]),
                     np.concatenate([gp_learner.y_pred - 1.96 * gp_learner.sigma,
                                     (gp_learner.y_pred + 1.96 * gp_learner.sigma)[::-1]]),
                     alpha=.5, fc='b', ec='None', label='95% confidence interval')
            plt.xlabel('bid')
            plt.ylabel('$cost(bid)$')
            plt.legend(loc='lower right')
            if i == self.n_obs - 1:
                plt.savefig("plots/bidding and learning (step 5)/cost_x_click_experiment_{}_estimation_month_{}.png".
                            format(self.exp, int(self.estimation_round/30)))
            plt.close(fig)
            """

        bid_value = np.array([bid])
        predicted_cost = gp_learner.gp.predict(bid_value.reshape(1, -1))
        # predict number of clicks with indicated bid
        float(predicted_cost)
        for i in self.arms:
            b = np.array([i]) # bid value
            pred_c_c = gp_learner.gp.predict(b.reshape(1, -1))
            self.temp_estimated_cost[i-1] = pred_c_c
        self.sigma_cost_per_click = gp_learner.sigma
        return predicted_cost

    def objective_function(self, price, bid, t):
        """This function returns the objective function, considering that conversion rate is known, while number of
        clicks and cost per click must be learnt as gaussian processes"""
        if t < 30:
            lambda_poisson = np.random.poisson(self.lambda_poisson, 1)
        else:
            resample = np.mean(self.lambda_values_poisson_months[0:t-29])
            lambda_poisson = np.random.poisson(resample, 1)
        self.lambda_values_poisson_months[t] = lambda_poisson
        #print(self.lambda_values_poisson_months[t])

        if t%120==0: # every 4 months a new gp is run
            reward = price * self.conv_rate[(price - 1)] * self.estimate_n_clicks(bid) * \
                     (1 + lambda_poisson) - (self.estimate_cost_x_click(bid) * self.estimate_n_clicks(bid))
        else:
            reward = price * self.conv_rate[(price - 1)] * self.temp_estimated_n_clicks[bid - 1] * \
                     (1 + lambda_poisson) - (self.temp_estimated_cost[bid - 1] *
                                                  self.temp_estimated_n_clicks[bid - 1])

        self.sigma_reward = (self.temp_estimated_n_clicks[bid - 1]**2 * self.sigma_cost_per_click[bid - 1]**2) + \
                            (price * self.conv_rate[(price - 1)] * (1 + self.lambda_poisson) -
                             self.temp_estimated_cost[bid - 1])**2 * self.sigma_n_clicks[bid - 1]**2 + \
                            (self.sigma_cost_per_click[bid - 1]**2 * self.sigma_n_clicks[bid - 1]**2)

        return reward

    def execute(self, scenario: Scenario, horizon, n_experiment):
        """This function executes the pricing experiment. The time horizon is 360, as we have to consider an horizon of
        1 year and the round is equal to one single day.

        horizon: time horizon of the experiment
        n_experiments: number of experiments to perform
        """
        ts_rewards_per_experiment = []  # store rewards

        self.result_table_ts = logger.create_table(horizon * n_experiment - 1, 5)
        # 6 columns in result table, first is the pulled arm by the algorithm, second is the reward of the algorithm,
        # third is the pulled arm by the clairvoyant
        # fourth is the reward of clairvoyant
        # fifth is the experiment
        # sixth is the time horizon

        self.params_ts = logger.create_table(horizon * n_experiment - 1, 3)
        # 6 columns in params table
        # first is the pulled arm by the algorithm
        # second is the reward of the algorithm
        # third is the alpha of parameter
        # fourth is the beta of parameter

        index = 0
        path_obj_ts = "results/step5/ts"
        path_p_ts = "results/step5/params_ts"

        for e in range(0, n_experiment):
            self.exp = e
            self.estimation_round = 1
            env = scenario.bidding_environment
            # simulate interaction
            ts_learner = TS_Learner(n_arms=self.n_arms)
            early_stopping = 0 #avoid infinite loop for safety constraint

            # iterate on number of rounds - simulate interaction between learner and environment
            for t in range(0, horizon):
                early_stopping = 0
                if t>0 & t%10==0:
                    # Number of estimation round determines the number of observations needed to compute the estimation
                    # of number of clicks and cost per click by the GP
                    self.estimation_round = self.estimation_round + 1
                    # The more the # rounds, the higher the number of observations

                if t > 10:
                    self.set_reward_successes()

                    # Thompson Sampling TS Learner
                    pulled_arm = ts_learner.pull_arm()  # learner compute arm to pull
                    reward = env.round(pulled_arm)  # environment compute reward given the pulled arm
                    price = 3
                    reward *= self.objective_function(price, self.arms[pulled_arm], t)
                    check_safety_constraint = - reward / self.sigma_reward
                    while check_safety_constraint > -0.84:
                        early_stopping = early_stopping + 1
                        pulled_arm = ts_learner.pull_arm()  # learner compute arm to pull
                        reward = env.round(pulled_arm)  # environment compute reward given the pulled arm
                        price = 3
                        reward *= self.objective_function(price, self.arms[pulled_arm], t)
                        check_safety_constraint = - reward / self.sigma_reward
                        if early_stopping == 10:
                            check_safety_constraint = -1

                    """if max(self.successes) >= 0.2:
                    VERSIONE VECCHIA PER VERIFICARE IL 20% DI PROB. DI SUCCESS.
                        while self.successes[pulled_arm] < 0.2:
                            pulled_arm = ts_learner.pull_arm()  # learner compute arm to pull"""
                else:
                    # Thompson Sampling TS Learner
                    pulled_arm = ts_learner.pull_arm()  # learner compute arm to pull
                    reward = env.round(pulled_arm)  # environment compute reward given the pulled arm
                    price = 3
                    reward *= self.objective_function(price, self.arms[pulled_arm], t)

                ts_learner.update(pulled_arm, reward)  # learner updates the rewards
                self.pulled_arms.append(self.arms[pulled_arm])

                # logging
                logger.update_table(self.result_table_ts, index, 0, pulled_arm)
                logger.update_table(self.result_table_ts, index, 1, reward)

                logger.update_table(self.params_ts, index, 0, pulled_arm)
                logger.update_table(self.params_ts, index, 1, reward)
                logger.update_table(self.params_ts, index, 2, ts_learner.beta_parameters[pulled_arm, 0])
                logger.update_table(self.params_ts, index, 3, ts_learner.beta_parameters[pulled_arm, 1])

                # Clairvoyant
                best_arm, best_reward = self.clairvoyant(t, price)
                # logging
                logger.update_table(self.result_table_ts, index, 2, best_arm)
                logger.update_table(self.result_table_ts, index, 3, best_reward)
                logger.update_table(self.result_table_ts, index, 4, e)
                logger.update_table(self.result_table_ts, index, 5, t)
                index = index + 1

                # Checks needed to verify reward of positive rewards of the pulled arms
                self.number_pulls_for_arm[pulled_arm] = self.number_pulls_for_arm[pulled_arm] + 1
                if reward > 0:
                    self.number_positive_rewards_for_arm[pulled_arm] = (self.number_positive_rewards_for_arm[pulled_arm]
                                                                        + 1)
                if t == 10:
                    self.set_probabilities(env)
            # store value of collected rewards
            ts_rewards_per_experiment.append(np.sum(ts_learner.collected_rewards))

        # select best arm TS
        self.best_arm = self.select_best_arm()

        print("The best arm (bid TS) is: " + str(self.best_arm))
        data = np.array(self.pulled_arms)
        labels = []
        for i in self.arms:
            count = np.count_nonzero(data==i)
            labels.append(count)

        fig, ax = plt.subplots(1, 1)
        #ax.hist(self.pulled_arms, histtype='bar', ec='blue')
        plt.bar(self.arms, height=labels)
        ax.set_title('Histogram of arms pulled - Bidding')
        ax.set_xlabel('Arms')
        ax.set_ylabel('Number of times each arm was pulled')
        ax.set_xticks(np.arange(self.n_arms)+1)
        ax.set_xticklabels(self.arms)
        rects = ax.patches
        best = self.best_arm

        for rect, label in zip(rects, labels):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, height + 5, label,
                    ha='center', va='bottom')
        plt.plot()
        plt.ylim(0, labels[best-1]+400)
        #plt.show()
        plt.savefig("plots/bidding and learning (step 5)/TS_for_bidding.png")

        logger.table_to_csv(self.result_table_ts, path_obj_ts)
        logger.table_to_csv(self.params_ts, path_p_ts)

    def select_best_arm(self):
        return max(set(self.pulled_arms), key=self.pulled_arms.count)

    def set_reward_successes(self):
        """Probabilities of the arms are updated according to the percentage of positive rewards that these arms had
        previously got"""
        for arm in range(self.n_arms):
            if self.number_pulls_for_arm[arm] > 0:
                self.successes[arm] = self.number_positive_rewards_for_arm[arm] / self.number_pulls_for_arm[arm]

    def set_probabilities(self, env: pricing_environment):
        "Change probabilities of env"
        """
        print("before: ")
        print(env.probabilities)
        for i in range(len(self.arms)):
            env.probabilities[i] = self.updated_probabilities[i]
        print("after: ")
        print(env.probabilities)"""
        s=1

    def clairvoyant(self, t, price):
        """This function returns the objective function, having a bid, number of clicks and cost per click fixed. The
        value that is changing is the price"""
        if t < 30:
            lambda_poisson = np.random.poisson(self.lambda_poisson, 1)
        else:
            resample = np.mean(self.lambda_values_poisson_months[0:t-29])
            lambda_poisson = np.random.poisson(resample, 1)
        self.lambda_values_poisson_months[t] = lambda_poisson

        rewards = list()
        for bid in self.arms:
            reward = price * self.conv_rate[(price - 1)] * self.temp_estimated_n_clicks[bid - 1] * \
                     (1 + lambda_poisson) - (self.temp_estimated_cost[bid - 1] *
                                             self.temp_estimated_n_clicks[bid - 1])
            rewards.append(reward)
        best_reward = max(rewards)
        best_arm = rewards.index(best_reward)
        return best_arm, best_reward