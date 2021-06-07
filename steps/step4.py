from dia.environments import Scenario
from dia.learner.pricing.pricing_learner import TS_Learner
from dia.environments.pricing.person_manager import Person_Manager
from dia.utils import logger

from operator import add

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Step4:
    """ This step is a context-generation task, data are considered as non aggregated, and we want to perform again the
    pricing task."""
    def __init__(self, probabilities, prices, features, categories, n_persons, number_clicks, cost_per_click,
                 lambda_poisson, arms, probabilities1, conv_rate):
        self.probabilities = probabilities
        self.prices = prices
        self.features = features
        self.categories = categories
        self.n_persons = n_persons
        self.single_split_reward = []

        self.evaluations_of_splitting = []
        """
        0: (0,1),2,3
        1: (0, 2), 1, 3
        2. (0, 3), 1, 2
        3: 0, (1, 2), 3
        4: 0, (1, 3), 2
        5: 0,1,(2,3)
        6: (0,1,2),3
        7: 0,(1,2,3)
        8: (0,1,2,3)
        9: 0,1,2,3
        10: (0,1),(2,3)
        """
        self.monthly_best = [] # 12 array list keeping best split per month

        # utilizziamo un person_manager per gestire la creazione di nuove persone
        self.p_manager = Person_Manager(self.categories, self.probabilities, self.features)
        self.result_table = logger.create_table(11, 3)
        # 3 columns in result table, first is the month, second is the best category, third is the number of categories
        # fourth is the reward of the best

        self.number_clicks = number_clicks
        self.cost_per_click = cost_per_click
        self.lambda_poisson = lambda_poisson

        self.arms = arms  # prices
        self.n_arms = len(self.arms)
        self.probabilities_ts = probabilities1  # probabilty for every arm
        self.conv_rate = conv_rate

        self.pulled_arms = []
        self.lambda_values_poisson_months = np.zeros(360)  # horizon one year

    def objective_function_split_4(self, poisson:list):
        """This function returns the objective function. Cost per click is now not useful as it would be the same
        for all the classes"""
        poisson_part = 0
        reward = 0
        temp = 0
        for j in range(len(self.categories)):
            partial = 0
            poisson_part = (poisson[j] + 1) * self.p_manager.categories_count[j]
            for i in range(len(self.prices)):
                partial += self.prices[i] * self.probabilities[j][i]
            temp = poisson_part * partial
            self.single_split_reward.append(temp)
            reward += temp
        reward -= 20 #penalty
        return reward

    def objective_function_unify(self, probability:list, number_buyers, poisson):
        """This function returns the objective function. Cost per click is now not useful as it would be the same
        for all the classes"""
        partial = 0
        for i in range(len(probability)):
            partial += self.prices[i] * probability[i]
        reward = partial * number_buyers * (1 + poisson)
        return reward

    def execute(self, scenario: Scenario, horizon, n_experiment):
        """This function executes the pricing experiment. The time horizon is 360, as we have to consider an horizon of
        1 year and the round is equal to one single day.

        horizon: time horizon of the experiment
        n_experiments: number of experiments to perform
        """
        ts_rewards_per_experiment = []  # store rewards
        path_obj = "results/step4/results"

        # Context Generation
        # Use a person_manager to manage the creation of new people
        for month in range(12):  # check every month for update
            persons = self.n_persons / 12
            for person in range(int(persons)):
                new_person = self.p_manager.new_person()
            # end of the month
            self.evaluation_split(scenario)
            max_value = max(self.evaluations_of_splitting)
            best_split = self.evaluations_of_splitting.index(max_value)
            print("month ", month)
            print(self.evaluations_of_splitting)
            self.monthly_best.append(best_split)
            logger.update_table(self.result_table, month, 0, month + 1)
            logger.update_table(self.result_table, month, 1, self.monthly_best[month])
            detail = self.get_splitting(best_split)
            number_cat = len(detail) - 1
            logger.update_table(self.result_table, month, 2, number_cat)
            logger.update_table(self.result_table, month, 3, max_value)
        print("months")
        print(self.monthly_best)
        #logger.table_to_csv(self.result_table, path_obj)

        def most_frequent(List):
            return max(set(List), key=List.count)

        final_best_split = most_frequent(self.monthly_best)

        final_categories = self.get_categories(final_best_split)

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
        path_obj_ts = "results/step4/ts"
        path_p_ts = "results/step4/params_ts"

        # THOMPSON SAMPLING
        for e in range(0, n_experiment):
            env = scenario.pricing_environment
            # simulate interaction
            ts_learner = TS_Learner(n_arms=self.n_arms)

            # iterate on number of rounds - simulate interaction between learner and environment
            for t in range(0, horizon):
                # Thompson Sampling TS Learner
                pulled_arm = ts_learner.pull_arm()  # learner compute arm to pull
                reward = env.round(pulled_arm)  # environment compute reward given the pulled arm
                reward *= self.objective_function(self.arms[pulled_arm], t)
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
                best_arm, best_reward = self.clairvoyant(t)
                # logging
                logger.update_table(self.result_table_ts, index, 2, best_arm)
                logger.update_table(self.result_table_ts, index, 3, best_reward)
                logger.update_table(self.result_table_ts, index, 4, e)
                logger.update_table(self.result_table_ts, index, 5, t)

                index = index + 1
            # store value of collected rewards
            ts_rewards_per_experiment.append(np.sum(ts_learner.collected_rewards))

            # select best arm TS
        self.best_arm = self.select_best_arm()
        # select best arm UCB1

        print("The best arm (price TS) is: " + str(self.best_arm))
        data = np.array(self.pulled_arms)
        labels = []
        for i in self.arms:
            count = np.count_nonzero(data == i)
            labels.append(count)

        fig, ax = plt.subplots(1, 1)
        # ax.hist(self.pulled_arms, histtype='bar', ec='blue')
        plt.bar(self.arms, height=labels)
        ax.set_title('Histogram of arms pulled - Pricing')
        ax.set_xlabel('Arms')
        ax.set_ylabel('Number of times each arm was pulled')
        ax.set_xticks(np.arange(self.n_arms) + 1)
        ax.set_xticklabels(self.arms)
        rects = ax.patches
        best = self.best_arm.astype(int)

        for rect, label in zip(rects, labels):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, height + 5, label,
                    ha='center', va='bottom')
        plt.plot()
        plt.ylim(0, labels[best - 1] + 300)
        # plt.show()
        plt.savefig("plots/pricing (step 4)/TS_for_pricing.png")
        plt.close(fig)

        logger.table_to_csv(self.result_table_ts, path_obj_ts)
        logger.table_to_csv(self.params_ts, path_p_ts)


            # simulate interaction
            #ts_learner = TS_Learner(n_arms=self.n_arms)

    def evaluation_split(self, scenario: Scenario):
        """Function that evaluates if splitting or not"""
        # evaluate splitting between couple
        poisson_list = [scenario.poisson1, scenario.poisson2, scenario.poisson2, scenario.poisson3]
        self.evaluations_of_splitting.clear()
        self.single_split_reward.clear()

        # reward with all classes split 1,2,3,4
        starting_point_reward = self.objective_function_split_4(poisson_list)

        for i in range(3): #index 0-2 for 3 experiments of unifying in couples
            for k in range(i+1, 4):
                overall_reward = 0
                poisson = (poisson_list[i] + poisson_list[k]) / 2
                probability = list(map(add, self.probabilities[i], self.probabilities[k]))
                probability[:] = [x / 2 for x in probability]
                number_persons = self.p_manager.categories_count[i] + self.p_manager.categories_count[k]

                reward_of_unified = self.objective_function_unify(probability, number_persons, poisson)

                left_cat = [0, 1, 2, 3]
                left_cat.remove(i)
                left_cat.remove(k)
                reward_still_split = 0
                for j in left_cat:
                    reward_still_split += self.single_split_reward[j]
                overall_reward = reward_still_split + reward_of_unified
                self.evaluations_of_splitting.append(overall_reward)

                """
                overall_reward = 0
                poisson = (poisson_list[i] + poisson_list[i+1]) / 2
                probability = list(map(add, self.probabilities[i], self.probabilities[i+1]))
                probability[:] = [x / 2 for x in probability]
                number_persons = self.p_manager.categories_count[i] + self.p_manager.categories_count[i+1]
    
                reward_of_unified = self.objective_function_unify(probability, number_persons, poisson)
    
                left_cat = [0, 1, 2, 3]
                left_cat.remove(i)
                left_cat.remove(i+1)
                reward_still_split = 0
                for j in left_cat:
                    reward_still_split += self.single_split_reward[j]
                overall_reward = reward_still_split + reward_of_unified
                self.evaluations_of_splitting.append(overall_reward)"""

        for i in range(2): #index 0-1 for 2 experiments of unifying in triples
            overall_reward = 0
            poisson = (poisson_list[i] + poisson_list[i + 1] + poisson_list[i + 2]) / 3
            probability = list(map(add, self.probabilities[i], self.probabilities[i + 1]))
            probability = list(map(add, probability, self.probabilities[i + 2]))
            probability[:] = [x / 3 for x in probability]
            number_persons = self.p_manager.categories_count[i] + self.p_manager.categories_count[i + 1] \
                             + self.p_manager.categories_count[i + 2]

            reward_of_unified = self.objective_function_unify(probability, number_persons, poisson)

            left_cat = [0, 1, 2, 3]
            left_cat.remove(i)
            left_cat.remove(i + 1)
            left_cat.remove(i + 2)
            reward_still_split = 0
            for j in left_cat:
                reward_still_split += self.single_split_reward[j]
            overall_reward = reward_still_split + reward_of_unified
            self.evaluations_of_splitting.append(overall_reward)

        # unify all
        for i in range(1): #case in which all classes have been unified
            overall_reward = 0
            poisson = (poisson_list[i] + poisson_list[i + 1] + poisson_list[i + 2] + poisson_list[i + 3]) / 4
            probability_1 = list(map(add, self.probabilities[i], self.probabilities[i + 1]))
            probability_2 = list(map(add, self.probabilities[i + 2], self.probabilities[i + 3]))
            probability = list(map(add, probability_1, probability_2))
            probability[:] = [x / 4 for x in probability]
            number_persons = self.p_manager.categories_count[i] + self.p_manager.categories_count[i + 1] \
                             + self.p_manager.categories_count[i + 2] + self.p_manager.categories_count[i + 3]

            reward_of_unified = self.objective_function_unify(probability, number_persons, poisson)
            self.evaluations_of_splitting.append(reward_of_unified)

        # case in which no split has been done
        self.evaluations_of_splitting.append(starting_point_reward)

        # unify 0-1 and 2-3
        for i in range(1):
            poisson_1 = (poisson_list[i] + poisson_list[i + 1]) / 2
            poisson_2 = (poisson_list[i + 2] + poisson_list[i + 3]) / 2
            probability_1 = list(map(add, self.probabilities[i], self.probabilities[i + 1]))
            probability_2 = list(map(add, self.probabilities[i + 2], self.probabilities[i + 3]))
            probability_1[:] = [x / 2 for x in probability_1]
            probability_2[:] = [x / 2 for x in probability_2]
            number_persons_1 = self.p_manager.categories_count[i] + self.p_manager.categories_count[i + 1]
            number_persons_2 = self.p_manager.categories_count[i + 2] + self.p_manager.categories_count[i + 3]

            reward_of_unified_1 = self.objective_function_unify(probability_1, number_persons_1, poisson_1)
            reward_of_unified_2 = self.objective_function_unify(probability_2, number_persons_2, poisson_2)
            overall_reward = reward_of_unified_1 + reward_of_unified_2
            self.evaluations_of_splitting.append(overall_reward)

    def get_categories(self, code):
        """
        0: (0, 1), 2, 3
        1: (0, 2), 1, 3
        2. (0, 3), 1, 2
        3: 0, (1, 2), 3
        4: 0, (1, 3), 2
        5: 0, 1, (2, 3)
        6: (0,1,2),3
        7: 0,(1,2,3)
        8: (0,1,2,3)
        9: 0,1,2,3
        10: (0,1),(2,3)
        """
        result = {}
        if code == 0:
            print("There are 3 contexts")
            result.update(number_classes=3)
            class_merged = self.categories[0:2]
            result.update(class_1=class_merged)
            result.update(class_2=self.categories[2])
            result.update(class_3=self.categories[3])
            print("class 1: ")
            print(result["class_1"])
            print("class 2: ")
            print(result["class_2"])
            print("class 3: ")
            print(result["class_3"])
        elif code == 1:
            print("There are 3 contexts")
            result.update(number_classes=3)
            class_merged = self.categories[0] + self.categories[2]
            result.update(class_1=class_merged)
            result.update(class_2=self.categories[1])
            result.update(class_3=self.categories[3])
            print("class 1: ")
            print(result["class_1"])
            print("class 2: ")
            print(result["class_2"])
            print("class 3: ")
            print(result["class_3"])
        elif code == 2:
            print("There are 3 contexts")
            result.update(number_classes=3)
            class_merged = self.categories[0] + self.categories[3]
            result.update(class_1=class_merged)
            result.update(class_2=self.categories[1])
            result.update(class_3=self.categories[2])
            print("class 1: ")
            print(result["class_1"])
            print("class 2: ")
            print(result["class_2"])
            print("class 3: ")
            print(result["class_3"])
        elif code == 3:
            print("There are 3 contexts")
            result.update(number_classes=3)
            class_merged = self.categories[1:3]
            result.update(class_1=self.categories[0])
            result.update(class_2=class_merged)
            result.update(class_3=self.categories[3])
            print("class 1: ")
            print(result["class_1"])
            print("class 2: ")
            print(result["class_2"])
            print("class 3: ")
            print(result["class_3"])
        elif code == 4:
            print("There are 3 contexts")
            result.update(number_classes=3)
            class_merged = self.categories[1] + self.categories[3]
            result.update(class_1=self.categories[0])
            result.update(class_2=class_merged)
            result.update(class_3=self.categories[2])
            print("class 1: ")
            print(result["class_1"])
            print("class 2: ")
            print(result["class_2"])
            print("class 3: ")
            print(result["class_3"])
        elif code == 5:
            print("There are 3 contexts")
            result.update(number_classes=3)
            class_merged = self.categories[2:4]
            result.update(class_1=self.categories[0])
            result.update(class_2=self.categories[1])
            result.update(class_3=class_merged)
            print("class 1: ")
            print(result["class_1"])
            print("class 2: ")
            print(result["class_2"])
            print("class 3: ")
            print(result["class_3"])
        elif code == 6:
            print("There are 2 contexts")
            result.update(number_classes=2)
            class_merged = self.categories[0:3]
            result.update(class_1=class_merged)
            result.update(class_2=self.categories[3])
            print("class 1: ")
            print(result["class_1"])
            print("class 2: ")
            print(result["class_2"])
        elif code == 7:
            print("There are 2 contexts")
            result.update(number_classes=2)
            class_merged = self.categories[1:4]
            result.update(class_1=self.categories[0])
            result.update(class_2=class_merged)
            print("class 1: ")
            print(result["class_1"])
            print("class 2: ")
            print(result["class_2"])
        elif code == 8:
            print("There is not any category")
            result.update(number_classes=1)
            class_merged = self.categories[0:4]
            result.update(single_class=class_merged)
            print("Only a single class: ")
            print(result["single_class"])
        elif code == 9:
            print("There are 4 contexts")
            result.update(number_classes=4)
            result.update(class_1=self.categories[0])
            result.update(class_2=self.categories[1])
            result.update(class_3=self.categories[2])
            result.update(class_4=self.categories[3])
            print("class 1: ")
            print(result["class_1"])
            print("class 2: ")
            print(result["class_2"])
            print("class 3: ")
            print(result["class_3"])
            print("class 4: ")
            print(result["class_3"])
        elif code == 10:
            print("There are 2 contexts")
            result.update(number_classes=2)
            class_merged_1 = self.categories[0:2]
            class_merged_2 = self.categories[2:4]
            result.update(class_1=class_merged_1)
            result.update(class_2=class_merged_2)
            print("class 1: ")
            print(result["class_1"])
            print("class 2: ")
            print(result["class_2"])

    def get_splitting(self, code):
        """
        0: (0, 1), 2, 3
        1: (0, 2), 1, 3
        2. (0, 3), 1, 2
        3: 0, (1, 2), 3
        4: 0, (1, 3), 2
        5: 0, 1, (2, 3)
        6: (0,1,2),3
        7: 0,(1,2,3)
        8: (0,1,2,3)
        9: 0,1,2,3
        10: (0,1),(2,3)
        """
        result = {}
        if code == 0:
            result.update(number_classes=3)
            class_merged = self.categories[0:2]
            result.update(class_1=class_merged)
            result.update(class_2=self.categories[2])
            result.update(class_3=self.categories[3])
            return result
        elif code == 1:
            result.update(number_classes=3)
            class_merged = self.categories[0] + self.categories[2]
            result.update(class_1=class_merged)
            result.update(class_2=self.categories[1])
            result.update(class_3=self.categories[3])
            return result
        elif code == 2:
            result.update(number_classes=3)
            class_merged = self.categories[0] + self.categories[3]
            result.update(class_1=class_merged)
            result.update(class_2=self.categories[1])
            result.update(class_3=self.categories[2])
            return result
        elif code == 3:
            result.update(number_classes=3)
            class_merged = self.categories[1:3]
            result.update(class_1=self.categories[0])
            result.update(class_2=class_merged)
            result.update(class_3=self.categories[3])
            return result
        elif code == 4:
            result.update(number_classes=3)
            class_merged = self.categories[1] + self.categories[3]
            result.update(class_1=self.categories[0])
            result.update(class_2=class_merged)
            result.update(class_3=self.categories[2])
            return result
        elif code == 5:
            print("There are 3 contexts")
            result.update(number_classes=3)
            class_merged = self.categories[2:4]
            result.update(class_1=self.categories[0])
            result.update(class_2=self.categories[1])
            result.update(class_3=class_merged)
            return result
        elif code == 6:
            result.update(number_classes=2)
            class_merged = self.categories[0:3]
            result.update(class_1=class_merged)
            result.update(class_2=self.categories[3])
            return result
        elif code == 7:
            result.update(number_classes=2)
            class_merged = self.categories[1:4]
            result.update(class_1=self.categories[0])
            result.update(class_2=class_merged)
            return result
        elif code == 8:
            result.update(number_classes=1)
            class_merged = self.categories[0:4]
            result.update(single_class=class_merged)
            return result
        elif code == 9:
            result.update(number_classes=4)
            result.update(class_1=self.categories[0])
            result.update(class_2=self.categories[1])
            result.update(class_3=self.categories[2])
            result.update(class_4=self.categories[3])
            return result
        elif code == 10:
            result.update(number_classes=2)
            class_merged_1 = self.categories[0:2]
            class_merged_2 = self.categories[2:4]
            result.update(class_1=class_merged_1)
            result.update(class_2=class_merged_2)
            return result

    def clairvoyant(self, t):
        """This function returns the objective function, having a bid, number of clicks and cost per click fixed. The
        value that is changing is the price"""
        if t < 30:
            lambda_poisson = np.random.poisson(self.lambda_poisson, 1)
        else:
            resample = np.mean(self.lambda_values_poisson_months[0:t-29])
            lambda_poisson = np.random.poisson(resample, 1)
        self.lambda_values_poisson_months[t] = lambda_poisson

        rewards = list()
        for price in self.arms:
            reward = price * self.conv_rate[(price - 1).astype(int)] * self.number_clicks * (1 + lambda_poisson) \
                     - (self.cost_per_click * self.number_clicks)
            rewards.append(reward)
        best_reward = max(rewards)
        best_arm = rewards.index(best_reward)
        return best_arm, best_reward

    def objective_function(self, price, t):
        """This function returns the objective function, having a bid, number of clicks and cost per click fixed. The
        value that is changing is the price"""
        if t < 30:
            lambda_poisson = np.random.poisson(self.lambda_poisson, 1)
        else:
            resample = np.mean(self.lambda_values_poisson_months[0:t-29])
            lambda_poisson = np.random.poisson(resample, 1)
        self.lambda_values_poisson_months[t] = lambda_poisson
        reward = price * self.conv_rate[(price - 1).astype(int)] * self.number_clicks * (1 + lambda_poisson) \
                 - (self.cost_per_click * self.number_clicks)
        return reward

    def select_best_arm(self):
        return max(set(self.pulled_arms), key=self.pulled_arms.count)

