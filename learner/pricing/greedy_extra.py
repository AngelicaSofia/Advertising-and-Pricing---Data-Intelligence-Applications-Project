class Greedy_Learner(Learner):
    '''Greedy Learner that has to compute at each round, the value of the expected reward for the pulled arm'''

    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.expected_rewards = np.zeros(n_arms)

    def pull_arm(self):
        """Greedy Learner selects the arm to pull at each round t, by maximizing the expected reward array.

        Note that we assume that each arm is pulled at least once, so in the first rounds all arms are pulled (arm 0 at
        round 0, arm 1 at round 1, till all arms are pulled once)."""

        if self.t < self.n_arms:
            return self.t
        # Select arm that maximizes the expected reward
        idxs = np.argwhere(self.expected_rewards == self.expected_rewards.max()).reshape(-1)
        # As we could have multiple indexes, we randomly select one of the index that maximizes the expected reward
        chosen_pulled_arm = np.random.choice(idxs)
        return chosen_pulled_arm

    def update(self, pulled_arm, reward):
        """Function that takes the pulled arm and the reward given by it"""
        self.t += 1
        self.update_observations(pulled_arm, reward)
        # Update array of expected rewards that is simply the average of collected expected rewards for this arm
        # Perform a recursive update fo the average
        self.expected_rewards[pulled_arm] = (self.expected_rewards[pulled_arm] * (self.t - 1) + reward) / self.t