from dia.environments import Scenario
from dia.environments import pricing_environment
from dia.learner.pricing.pricing_learner import TS_Learner
from dia.learner.advertising.gp_learner import GP_Learner
from dia.environments import function_manager as fm
from dia.utils import logger

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

class Step7:
    """ This step is a pure ... TODO"""
    def __init__(self, lambda_poisson, arms, conv_rate, n_obs, noise_std_n_clicks,
                 noise_std_cost_x_click):
        self.lambda_poisson_1 = lambda_poisson*5/3 #estimate lambda for class 1
        self.lambda_poisson_2 = lambda_poisson #estimate lambda for class 2
        self.lambda_poisson_3 = lambda_poisson*1/3 #estimate lambda for class 3

        self.lambda_values_poisson_1_months = np.zeros(360)  # horizon one year
        self.lambda_values_poisson_2_months = np.zeros(360)  # horizon one year
        self.lambda_values_poisson_3_months = np.zeros(360)  # horizon one year

        self.bid_arms = arms #bids
        self.price_arms = arms  # prices
        self.n_bid_arms = len(self.bid_arms)
        self.n_price_arms = len(self.price_arms)
        self.conv_rate_class_1 = conv_rate[0]
        self.conv_rate_class_2 = conv_rate[1]
        self.conv_rate_class_3 = conv_rate[2]
        self.n_obs = n_obs
        self.noise_std_n_clicks = noise_std_n_clicks
        self.noise_std_cost_x_click = noise_std_cost_x_click
        self.estimation_round = 1
        self.exp = 0 # experiment
        # estimation of functions is not done every round, but every ten rounds.
        self.temp_estimated_cost = [0] * 10
        self.temp_estimated_n_clicks_class_1 = [0] * 10
        self.temp_estimated_n_clicks_class_2 = [0] * 10
        self.temp_estimated_n_clicks_class_3 = [0] * 10

        self.pulled_bid_arms = []
        self.pulled_price_arms = []
        # Lists to verify that the reward for the arm is positive
        self.number_pulls_for_bid_arm = [0] * len(self.bid_arms)
        self.number_positive_rewards_for_bid_arm = [0] * len(self.bid_arms)
        self.number_pulls_for_price_arm = [0] * len(self.price_arms)
        self.number_positive_rewards_for_price_arm = [0] * len(self.price_arms)

        self.sigma_reward = 0
        self.sigma_cost_per_click = np.zeros(len(self.bid_arms))
        self.sigma_n_clicks_class_1 = np.zeros(len(self.bid_arms))
        self.sigma_n_clicks_class_2 = np.zeros(len(self.bid_arms))
        self.sigma_n_clicks_class_3 = np.zeros(len(self.bid_arms))

    def generate_observations(self, f_to_estimate, x, customer_class):
        """Function to generate observations. Considering number of clicks and cost per click, the function depends on
        the bid (x == bid)."""
        if f_to_estimate == "n_clicks":
            # x = bid
            n = fm.get_n_click(x, customer_class)
            # Division in classes is known
            return n + np.random.normal(0, self.noise_std_n_clicks, size=n.shape)
        if f_to_estimate == "cost_per_click":
            # x = bid
            return fm.get_cost_x_click(x) + np.random.normal(0, self.noise_std_cost_x_click,
                                                             size=fm.get_cost_x_click(x).shape)
        return

    def estimate_n_clicks(self, bid, customer_class):
        gp_learner = GP_Learner(self.noise_std_n_clicks)
        x_obs = np.array([])
        y_obs = np.array([])
        for i in range(0, self.n_obs):
            gp_learner = GP_Learner(self.noise_std_n_clicks)

            new_x_obs = np.random.choice(self.bid_arms, 1)
            # 1 is the customer class
            new_y_obs = self.generate_observations("n_clicks", new_x_obs, customer_class)

            x_obs = np.append(x_obs, new_x_obs.astype(float))
            y_obs = np.append(y_obs, new_y_obs.astype(float))

            X = np.atleast_2d(x_obs).T
            Y = y_obs.ravel()

            gp_learner.learn(X, Y, self.bid_arms)

            fig = plt.figure(i + self.n_obs * (self.estimation_round) / 10)
            plt.plot(gp_learner.x_pred, fm.get_n_click(gp_learner.x_pred, customer_class), 'r', label=r'$n clicks(bid) class {}$'.
                     format(customer_class))
            plt.plot(X.ravel(), Y, 'ro', label=u'observed clicks')
            plt.plot(gp_learner.x_pred, gp_learner.y_pred, 'b-', label=u'predicted clicks')
            plt.fill(np.concatenate([gp_learner.x_pred, gp_learner.x_pred[::-1]]),
                     np.concatenate([gp_learner.y_pred - 1.96 * gp_learner.sigma,
                                     (gp_learner.y_pred + 1.96 * gp_learner.sigma)[::-1]]),
                     alpha=.5, fc='b', ec='None', label='95% confidence interval')
            plt.xlabel('$bid$')
            plt.ylabel('$n_clicks(bid)$')
            plt.legend(loc='lower right')
            # plt.show()
            if i == self.n_obs - 1:
                plt.savefig(
                    "plots/bidding and pricing (step 7)/n_clicks_experiment_class_{}_{}_estimation_month_{}.png".
                    format(customer_class, self.exp, int(self.estimation_round / 30)))
            plt.close(fig)


        # saving temporal values for the estimations
        bid_value = np.array([bid])
        predicted_n_clicks = gp_learner.gp.predict(bid_value.reshape(1, -1))
        # predict number of clicks with indicated bid
        float(predicted_n_clicks)
        if customer_class == 1:
            for i in self.bid_arms:
                b = np.array([i])  # bid value
                pred_n_c = gp_learner.gp.predict(b.reshape(1, -1))
                self.temp_estimated_n_clicks_class_1[i - 1] = pred_n_c
            self.sigma_n_clicks_class_1 = gp_learner.sigma
        if customer_class == 2:
            for i in self.bid_arms:
                b = np.array([i])  # bid value
                pred_n_c = gp_learner.gp.predict(b.reshape(1, -1))
                self.temp_estimated_n_clicks_class_2[i - 1] = pred_n_c
            self.sigma_n_clicks_class_2 = gp_learner.sigma
        if customer_class == 3:
            for i in self.bid_arms:
                b = np.array([i])  # bid value
                pred_n_c = gp_learner.gp.predict(b.reshape(1, -1))
                self.temp_estimated_n_clicks_class_3[i - 1] = pred_n_c
            self.sigma_n_clicks_class_3 = gp_learner.sigma

        return predicted_n_clicks

    def estimate_cost_x_click(self, bid):
        gp_learner = GP_Learner(self.noise_std_n_clicks)
        x_obs = np.array([])
        y_obs = np.array([])
        for i in range(0, self.n_obs):
            gp_learner = GP_Learner(self.noise_std_cost_x_click)

            new_x_obs = np.random.choice(self.bid_arms, 1)
            new_y_obs = self.generate_observations("cost_per_click", new_x_obs, 1) # 1 for customer_class
            # customer class is not needed for cost per click

            x_obs = np.append(x_obs, new_x_obs.astype(float))
            y_obs = np.append(y_obs, new_y_obs.astype(float))

            X = np.atleast_2d(x_obs).T
            Y = y_obs.ravel()

            gp_learner.learn(X, Y, self.bid_arms)
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
        for i in self.bid_arms:
            b = np.array([i]) # bid value
            pred_c_c = gp_learner.gp.predict(b.reshape(1, -1))
            self.temp_estimated_cost[i-1] = pred_c_c
        self.sigma_cost_per_click = gp_learner.sigma
        return predicted_cost

    def delay_lambda_poisson(self, t):
        if t < 30:
            lambda_poisson_1 = np.random.poisson(self.lambda_poisson_1, 1)
            self.lambda_poisson_1 = lambda_poisson_1 # make update in order not have problems with previous written code
            # delay has been inserted later
            lambda_poisson_2 = np.random.poisson(self.lambda_poisson_2, 1)
            self.lambda_poisson_2 = lambda_poisson_2
            lambda_poisson_3 = np.random.poisson(self.lambda_poisson_3, 1)
            self.lambda_poisson_3 = lambda_poisson_3
        else:
            resample_1 = np.mean(self.lambda_values_poisson_1_months[0:t-29])
            lambda_poisson_1 = np.random.poisson(resample_1, 1)
            resample_2 = np.mean(self.lambda_values_poisson_2_months[0:t - 29])
            lambda_poisson_2 = np.random.poisson(resample_2, 1)
            resample_3 = np.mean(self.lambda_values_poisson_3_months[0:t - 29])
            lambda_poisson_3 = np.random.poisson(resample_3, 1)
            self.lambda_poisson_1 = lambda_poisson_1
            self.lambda_poisson_2 = lambda_poisson_2
            self.lambda_poisson_3 = lambda_poisson_3

        self.lambda_values_poisson_1_months[t] = lambda_poisson_1
        self.lambda_values_poisson_2_months[t] = lambda_poisson_2
        self.lambda_values_poisson_3_months[t] = lambda_poisson_3

    def objective_function(self, price, bid, t):
        """This function returns the objective function, considering that conversion rate is known, while number of
        clicks and cost per click must be learnt as gaussian processes"""
        self.delay_lambda_poisson(t) #Update the lambda of different classes according to the delay

        if t%120==0: # every 4 months a new gp is run
            part1 = (price *
                     self.conv_rate_class_1[(price - 1)] *
                     self.estimate_n_clicks(bid, 1) *
                     (1 + self.lambda_poisson_1)) - (self.estimate_cost_x_click(bid) *
                                                     self.estimate_n_clicks(bid, 1))
            part2 = (price *
                     self.conv_rate_class_2[(price - 1)] *
                     self.estimate_n_clicks(bid, 2) *
                     (1 + self.lambda_poisson_2)) - (self.estimate_cost_x_click(bid) *
                                                     self.estimate_n_clicks(bid, 2))
            part3 = (price *
                     self.conv_rate_class_3[(price - 1)]*
                     self.estimate_n_clicks(bid, 3) *
                     (1 + self.lambda_poisson_3)) - (self.estimate_cost_x_click(bid) *
                                                     self.estimate_n_clicks(bid, 3))

        else:
            part1 = (price *
                     self.conv_rate_class_1[(price - 1)]*
                     self.temp_estimated_n_clicks_class_1[bid - 1] *
                     (1 + self.lambda_poisson_1)) - (self.temp_estimated_cost[bid - 1] *
                                                     self.temp_estimated_n_clicks_class_1[bid - 1])
            part2 = (price *
                     self.conv_rate_class_2[(price - 1)] *
                     self.temp_estimated_n_clicks_class_2[bid - 1] *
                     (1 + self.lambda_poisson_2)) - (self.temp_estimated_cost[bid - 1] *
                                                     self.temp_estimated_n_clicks_class_2[bid - 1])
            part3 = (price *
                     self.conv_rate_class_3[(price - 1)] *
                     self.temp_estimated_n_clicks_class_3[bid - 1] *
                     (1 + self.lambda_poisson_3)) - (self.temp_estimated_cost[bid - 1] *
                                                     self.temp_estimated_n_clicks_class_3[bid - 1])

        reward = part1 + part2 + part3
        temp_sigma_1 = (self.temp_estimated_n_clicks_class_1[bid - 1]**2 * self.sigma_cost_per_click[bid - 1]**2) + \
                       (price * self.conv_rate_class_1[(price - 1)] * (1 + self.lambda_poisson_1) -
                        self.temp_estimated_cost[bid - 1])**2 * self.sigma_n_clicks_class_1[bid - 1]**2 + \
                       (self.sigma_cost_per_click[bid - 1]**2 * self.sigma_n_clicks_class_1[bid - 1]**2)

        temp_sigma_2 = (self.temp_estimated_n_clicks_class_2[bid - 1] ** 2 * self.sigma_cost_per_click[bid - 1] ** 2) + \
                       (price * self.conv_rate_class_2[(price - 1)] * (1 + self.lambda_poisson_2) -
                        self.temp_estimated_cost[bid - 1]) ** 2 * self.sigma_n_clicks_class_2[bid - 1] ** 2 + \
                       (self.sigma_cost_per_click[bid - 1] ** 2 * self.sigma_n_clicks_class_2[bid - 1] ** 2)

        temp_sigma_3 = (self.temp_estimated_n_clicks_class_3[bid - 1] ** 2 * self.sigma_cost_per_click[bid - 1] ** 2) + \
                       (price * self.conv_rate_class_3[(price - 1)] * (1 + self.lambda_poisson_3) -
                        self.temp_estimated_cost[bid - 1]) ** 2 * self.sigma_n_clicks_class_3[bid - 1] ** 2 + \
                       (self.sigma_cost_per_click[bid - 1] ** 2 * self.sigma_n_clicks_class_3[bid - 1] ** 2)

        self.sigma_reward = (temp_sigma_1 + temp_sigma_2 + temp_sigma_3)/3

        return reward

    def execute(self, scenario: Scenario, horizon, n_experiment):
        """This function executes the pricing experiment. The time horizon is 360, as we have to consider an horizon of
        1 year and the round is equal to one single day.

        horizon: time horizon of the experiment
        n_experiments: number of experiments to perform
        """
        ts_bid_rewards_per_experiment = []  # store rewards bidding
        ts_price_rewards_per_experiment = []  # store rewards pricing

        ts_bid_rewards_per_experiment = []  # store rewards bidding
        ts_price_rewards_per_experiment = []  # store rewards pricing

        self.result_table_ts_price = logger.create_table(horizon * n_experiment - 1, 5)
        # 6 columns in result table, first is the pulled arm by the algorithm, second is the reward of the algorithm,
        # third is the pulled arm by the clairvoyant
        # fourth is the reward of clairvoyant
        # fifth is the experiment
        # sixth is the time horizon

        self.params_ts_price = logger.create_table(horizon * n_experiment - 1, 3)
        # 6 columns in params table
        # first is the pulled arm by the algorithm
        # second is the reward of the algorithm
        # third is the alpha of parameter
        # fourth is the beta of parameter

        index = 0
        path_obj_ts_price = "results/step7/ts_price"
        path_p_ts_price = "results/step7/params_ts_price"

        self.result_table_ts_bid = logger.create_table(horizon * n_experiment - 1, 5)
        # 6 columns in result table, first is the pulled arm by the algorithm, second is the reward of the algorithm,
        # third is the pulled arm by the clairvoyant
        # fourth is the reward of clairvoyant
        # fifth is the experiment
        # sixth is the time horizon

        self.params_ts_bid = logger.create_table(horizon * n_experiment - 1, 3)
        # 6 columns in params table
        # first is the pulled arm by the algorithm
        # second is the reward of the algorithm
        # third is the alpha of parameter
        # fourth is the beta of parameter

        path_obj_ts_bid = "results/step7/ts_bid"
        path_p_ts_bid = "results/step7/params_ts_bid"

        for e in range(0, n_experiment):
            self.exp = e
            self.estimation_round = 1
            bidding_env = scenario.joint_classes_bidding_environment
            pricing_env = scenario.joint_classes_pricing_environment

            # simulate interaction
            ts_bid_learner = TS_Learner(n_arms=self.n_bid_arms)
            ts_price_learner = TS_Learner(n_arms=self.n_price_arms)
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

                    # Thompson Sampling TS Learner for bidding
                    pulled_bid_arm = ts_bid_learner.pull_arm()  # learner compute arm to pull
                    bid_reward = bidding_env.round(pulled_bid_arm)
                    # bidding environment compute reward given the pulled arm

                    # Thompson Sampling TS Learner for pricing
                    pulled_price_arm = ts_price_learner.pull_arm()  # learner compute arm to pull
                    price_reward = pricing_env.round(pulled_price_arm)
                    # bidding environment compute reward given the pulled arm

                    #price = random.randint(1, 10)
                    multiplier = self.objective_function(self.price_arms[pulled_price_arm],
                                                         self.bid_arms[pulled_bid_arm],
                                                         t)

                    bid_reward *= multiplier
                    price_reward *= multiplier

                    check_safety_constraint = - bid_reward / self.sigma_reward
                    while check_safety_constraint > -0.84:
                        print("early stopping: " + str(early_stopping))
                        print("safety value: " + str(check_safety_constraint))
                        early_stopping = early_stopping + 1
                        pulled_bid_arm = ts_bid_learner.pull_arm()  # learner compute arm to pull
                        bid_reward = bidding_env.round(pulled_bid_arm)  # environment compute reward given pulled arm
                        #reward = (bid_reward + price_reward) / 2
                        #price = random.randint(1, 10)
                        #reward *= self.objective_function(self.price_arms[pulled_price_arm],
                        #                                  self.bid_arms[pulled_bid_arm],
                        #                                  t)
                        multiplier = self.objective_function(self.price_arms[pulled_price_arm],
                                                             self.bid_arms[pulled_bid_arm],
                                                             t)
                        bid_reward *= multiplier
                        check_safety_constraint = - bid_reward / self.sigma_reward
                        if early_stopping == 10:
                            check_safety_constraint = -1

                else:
                    # Thompson Sampling TS Learner for bidding
                    pulled_bid_arm = ts_bid_learner.pull_arm()  # learner compute arm to pull
                    bid_reward = bidding_env.round(pulled_bid_arm)
                    # bidding environment compute reward given the pulled arm

                    # Thompson Sampling TS Learner for pricing
                    pulled_price_arm = ts_price_learner.pull_arm()  # learner compute arm to pull
                    price_reward = pricing_env.round(pulled_price_arm)
                    # bidding environment compute reward given the pulled arm

                    #reward = (bid_reward + price_reward) / 2
                    # price = random.randint(1, 10)

                    #reward *= self.objective_function(self.price_arms[pulled_price_arm],
                    #                                  self.bid_arms[pulled_bid_arm],
                    #                                  t)
                    bid_reward *= self.objective_function(self.price_arms[pulled_price_arm],
                                                          self.bid_arms[pulled_bid_arm],
                                                          t)
                    price_reward *= self.objective_function(self.price_arms[pulled_price_arm],
                                                            self.bid_arms[pulled_bid_arm],
                                                            t)

                ts_bid_learner.update(pulled_bid_arm, bid_reward)  # bid learner updates the rewards
                ts_price_learner.update(pulled_price_arm, price_reward)  # price learner updates the rewards

                self.pulled_bid_arms.append(self.bid_arms[pulled_bid_arm])
                self.pulled_price_arms.append(self.price_arms[pulled_price_arm])

                #####################################################################################################
                # logging
                logger.update_table(self.result_table_ts_bid, index, 0, pulled_bid_arm)
                logger.update_table(self.result_table_ts_bid, index, 1, bid_reward)
                logger.update_table(self.result_table_ts_price, index, 0, pulled_price_arm)
                logger.update_table(self.result_table_ts_price, index, 1, price_reward)

                logger.update_table(self.params_ts_bid, index, 0, pulled_bid_arm)
                logger.update_table(self.params_ts_bid, index, 1, bid_reward)
                logger.update_table(self.params_ts_bid, index, 2, ts_bid_learner.beta_parameters[pulled_bid_arm, 0])
                logger.update_table(self.params_ts_bid, index, 3, ts_bid_learner.beta_parameters[pulled_bid_arm, 1])
                logger.update_table(self.params_ts_price, index, 0, pulled_price_arm)
                logger.update_table(self.params_ts_price, index, 1, price_reward)
                logger.update_table(self.params_ts_price, index, 2, ts_price_learner.beta_parameters[pulled_price_arm,
                                                                                                     0])
                logger.update_table(self.params_ts_price, index, 3, ts_price_learner.beta_parameters[pulled_price_arm,
                                                                                                     1])

                # Clairvoyant
                best_arm_bid, best_reward_bid = self.clairvoyant_bid(t, pulled_price_arm)
                best_arm_price, best_reward_price = self.clairvoyant_price(t, pulled_bid_arm)
                # logging
                logger.update_table(self.result_table_ts_bid, index, 2, best_arm_bid)
                logger.update_table(self.result_table_ts_bid, index, 3, best_reward_bid)
                logger.update_table(self.result_table_ts_bid, index, 4, e)
                logger.update_table(self.result_table_ts_bid, index, 5, t)
                logger.update_table(self.result_table_ts_price, index, 2, best_arm_price)
                logger.update_table(self.result_table_ts_price, index, 3, best_reward_price)
                logger.update_table(self.result_table_ts_price, index, 4, e)
                logger.update_table(self.result_table_ts_price, index, 5, t)
                index = index + 1
                #####################################################################################################

                # Checks needed to verify reward of positive rewards of the pulled arms
                self.number_pulls_for_bid_arm[pulled_bid_arm] = self.number_pulls_for_bid_arm[pulled_bid_arm] + 1
                if bid_reward > 0:
                    self.number_positive_rewards_for_bid_arm[pulled_bid_arm] = (
                            self.number_positive_rewards_for_bid_arm[pulled_bid_arm] + 1)

            # store value of collected rewards
            ts_bid_rewards_per_experiment.append(np.sum(ts_bid_learner.collected_rewards))
            ts_price_rewards_per_experiment.append(np.sum(ts_price_learner.collected_rewards))

        # select best arm TS bidding
        self.best_bid_arm = self.select_best_bid_arm()
        # select best arm TS pricing
        self.best_price_arm = self.select_best_price_arm()

        print("The best arm (bid TS) is: " + str(self.best_bid_arm))
        print("The best arm (price TS) is: " + str(self.best_price_arm))
        # TS for bidding plot
        data = np.array(self.pulled_bid_arms)
        labels = []
        for i in self.bid_arms:
            count = np.count_nonzero(data==i)
            labels.append(count)

        fig, ax = plt.subplots(1, 1)
        #ax.hist(self.pulled_bid_arms, histtype='bar', ec='blue')
        plt.bar(self.bid_arms, height=labels)
        ax.set_title('Histogram of arms pulled - Bidding')
        ax.set_xlabel('Arms')
        ax.set_ylabel('Number of times each arm was pulled')
        ax.set_xticks(np.arange(self.n_bid_arms)+1)
        ax.set_xticklabels(self.bid_arms)
        rects = ax.patches
        best_bid = self.best_bid_arm

        for rect, label in zip(rects, labels):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, height + 5, label,
                    ha='center', va='bottom')
        plt.plot()
        plt.ylim(0, labels[best_bid-1]+300)
        #plt.show()
        plt.savefig("plots/TS_for_bidding.png")

        # TS for pricing plot
        data_price = np.array(self.pulled_price_arms)
        labels_price = []
        for i in self.price_arms:
            count_price = np.count_nonzero(data_price == i)
            labels_price.append(count_price)

        fig, ax = plt.subplots(1, 1)
        # ax.hist(self.pulled_bid_arms, histtype='bar', ec='blue')
        plt.bar(self.price_arms, height=labels_price)
        ax.set_title('Histogram of arms pulled - Pricing')
        ax.set_xlabel('Arms')
        ax.set_ylabel('Number of times each arm was pulled')
        ax.set_xticks(np.arange(self.n_price_arms) + 1)
        ax.set_xticklabels(self.price_arms)
        rects = ax.patches
        best_price = self.best_price_arm

        for rect, label in zip(rects, labels_price):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, height + 5, label,
                    ha='center', va='bottom')
        plt.plot()
        plt.ylim(0, labels_price[best_price - 1] + 300)
        # plt.show()
        plt.savefig("plots/TS_for_pricing.png")

        logger.table_to_csv(self.result_table_ts_price, path_obj_ts_price)
        logger.table_to_csv(self.params_ts_price, path_p_ts_price)
        logger.table_to_csv(self.result_table_ts_bid, path_obj_ts_bid)
        logger.table_to_csv(self.params_ts_bid, path_p_ts_bid)

    def select_best_bid_arm(self):
        return max(set(self.pulled_bid_arms), key=self.pulled_bid_arms.count)

    def select_best_price_arm(self):
        return max(set(self.pulled_price_arms), key=self.pulled_price_arms.count)

    def clairvoyant_bid(self, t, price):
        """This function returns the objective function, having a bid, number of clicks and cost per click fixed. The
        value that is changing is the price"""
        rewards = list()
        for bid in self.bid_arms:
            part1 = (price *
                     self.conv_rate_class_1[(price - 1)] *
                     self.temp_estimated_n_clicks_class_1[bid - 1] *
                     (1 + self.lambda_poisson_1)) - (self.temp_estimated_cost[bid - 1] *
                                                     self.temp_estimated_n_clicks_class_1[bid - 1])
            part2 = (price *
                     self.conv_rate_class_2[(price - 1)] *
                     self.temp_estimated_n_clicks_class_2[bid - 1] *
                     (1 + self.lambda_poisson_2)) - (self.temp_estimated_cost[bid - 1] *
                                                     self.temp_estimated_n_clicks_class_2[bid - 1])
            part3 = (price *
                     self.conv_rate_class_3[(price - 1)] *
                     self.temp_estimated_n_clicks_class_3[bid - 1] *
                     (1 + self.lambda_poisson_3)) - (self.temp_estimated_cost[bid - 1] *
                                                     self.temp_estimated_n_clicks_class_3[bid - 1])

            reward = part1 + part2 + part3
            rewards.append(reward)
        best_reward = max(rewards)
        best_arm = rewards.index(best_reward)
        return best_arm, best_reward

    def clairvoyant_price(self, t, bid):
        """This function returns the objective function, having a bid, number of clicks and cost per click fixed. The
        value that is changing is the price"""
        rewards = list()
        for price in self.price_arms:
            part1 = (price *
                     self.conv_rate_class_1[(price - 1)] *
                     self.temp_estimated_n_clicks_class_1[bid - 1] *
                     (1 + self.lambda_poisson_1)) - (self.temp_estimated_cost[bid - 1] *
                                                     self.temp_estimated_n_clicks_class_1[bid - 1])
            part2 = (price *
                     self.conv_rate_class_2[(price - 1)] *
                     self.temp_estimated_n_clicks_class_2[bid - 1] *
                     (1 + self.lambda_poisson_2)) - (self.temp_estimated_cost[bid - 1] *
                                                     self.temp_estimated_n_clicks_class_2[bid - 1])
            part3 = (price *
                     self.conv_rate_class_3[(price - 1)] *
                     self.temp_estimated_n_clicks_class_3[bid - 1] *
                     (1 + self.lambda_poisson_3)) - (self.temp_estimated_cost[bid - 1] *
                                                     self.temp_estimated_n_clicks_class_3[bid - 1])

            reward = part1 + part2 + part3
            rewards.append(reward)
        best_reward = max(rewards)
        best_arm = rewards.index(best_reward)
        return best_arm, best_reward


