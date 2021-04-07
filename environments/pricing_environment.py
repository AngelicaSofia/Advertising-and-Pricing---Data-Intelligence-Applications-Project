import numpy as np

class pricing_environment():
    """
    Environment with information about the candidates and their probabilities
    """

    def __init__(self, n_arms, probabilities):
        self.n_arms = n_arms
        self.probabilities = probabilities

    def round(self, pulled_arm):
        # returns the reward of the arm chosen
        p = self.probabilities[pulled_arm]
        reward = np.random.binomial(1, p)
        return reward