from dia.environments import Scenario
from dia.learner.pricing.pricing_learner import TS_Learner
from dia.environments.pricing.person_manager import Person_Manager

from operator import add

class Step4:
    """ This step is a context-generation task, data are considered as non aggregated, and we want to perform again the
    pricing task."""
    def __init__(self, probabilities, prices, features, categories, n_persons):
        self.probabilities = probabilities
        self.prices = prices
        self.features = features
        self.categories = categories
        self.n_persons = n_persons
        self.single_split_reward = []

        self.evaluations_of_splitting = []
        """
        0: (0,1),2,3
        1: 0,(1,2),3
        2: 0,1,(2,3)
        3: (0,1,2),3
        4: 0,(1,2,3)
        5: (0,1,2,3)
        6: 0,1,2,3
        7: (0,1),(2,3)
        """
        self.monthly_best = [] # 12 array list keeping best split per month

        # utilizziamo un person_manager per gestire la creazione di nuove persone
        self.p_manager = Person_Manager(self.categories, self.probabilities, self.features)


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
        print("months")
        print(self.monthly_best)

        def most_frequent(List):
            return max(set(List), key=List.count)

        final_best_split = most_frequent(self.monthly_best)

        final_categories = self.get_categories(final_best_split)

        for e in range(0, n_experiment):
            env = scenario.pricing_environment


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
            self.evaluations_of_splitting.append(overall_reward)

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
        0: (0,1),2,3
        1: 0,(1,2),3
        2: 0,1,(2,3)
        3: (0,1,2),3
        4: 0,(1,2,3)
        5: (0,1,2,3)
        6: 0,1,2,3
        """
        result = {}
        if code == 0:
            print("There are 3 categories")
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
            print("There are 3 categories")
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
        elif code == 2:
            print("There are 3 categories")
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
        elif code == 3:
            print("There are 2 categories")
            result.update(number_classes=2)
            class_merged = self.categories[0:3]
            result.update(class_1=class_merged)
            result.update(class_2=self.categories[3])
            print("class 1: ")
            print(result["class_1"])
            print("class 2: ")
            print(result["class_2"])
        elif code == 4:
            print("There are 2 categories")
            result.update(number_classes=2)
            class_merged = self.categories[1:4]
            result.update(class_1=self.categories[0])
            result.update(class_2=class_merged)
            print("class 1: ")
            print(result["class_1"])
            print("class 2: ")
            print(result["class_2"])
        elif code == 5:
            print("There is not any category")
            result.update(number_classes=1)
            class_merged = self.categories[0:4]
            result.update(single_class=class_merged)
            print("Only a single class: ")
            print(result["single_class"])
        elif code == 6:
            print("There are 4 categories")
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
        elif code == 7:
            print("There are 2 categories")
            result.update(number_classes=2)
            class_merged_1 = self.categories[0:2]
            class_merged_2 = self.categories[2:4]
            result.update(class_1=class_merged_1)
            result.update(class_2=class_merged_2)
            print("class 1: ")
            print(result["class_1"])
            print("class 2: ")
            print(result["class_2"])
