import numpy as np

class Step3:
    def __init__(self):
        self.core = 0

    def execute(self):
        n_arms = 4
        p = np.array([0.15, 0.1, 0.1, 0.35])  # bernoulli distribution
        opt = p[3]

        T = 30  # horizon
        n_experiment = 10
        ts_rewards_per_experiment = []  # store rewards
        gr_rewards_per_experiment = []  # store rewards

        for e in range(0, n_experiment):
            env = pricing_environment(n_arms=n_arms, probabilities=p)
            # simulate interaction
            ts_learner = TS_Learner(n_arms=n_arms)
            gr_learner = Greedy_Learner(n_arms=n_arms)

            # iterate on number of rounds - simulate interaction between learner and environment
            for t in range(0, T):
                # Thompson Sampling TS Learner
                pulled_arm = ts_learner.pull_arm()  # learner compute arm to pull
                reward = env.round(pulled_arm)  # environment compute reward given the pulled arm
                ts_learner.update(pulled_arm, reward)  # learner updates the rewards

                # Greedy Learner GR Learner
                pulled_arm = gr_learner.pull_arm()  # learner compute arm to pull
                reward = env.round(pulled_arm)  # environment compute reward given the pulled arm
                gr_learner.update(pulled_arm, reward)  # learner updates the rewards

            # store value fo collected rewards
            ts_rewards_per_experiment.append(np.sum(ts_learner.collected_rewards))
            gr_rewards_per_experiment.append(np.sum(gr_learner.collected_rewards))

        plt.figure(0)
        plt.xlabel("t")
        plt.ylabel("Regret")
        # regret is the cumulative sum of the difference between the optimum and the collected reward by agent
        plt.plot(np.cumsum(np.mean(opt - ts_rewards_per_experiment, axis=0)), 'r')
        plt.plot(np.cumsum(np.mean(opt - gr_rewards_per_experiment, axis=0)), 'g')
        plt.legend("TS", "Greedy")
        plt.show()