import numpy as np
import matplotlib.pyplot as plt

from dia.environments import function_manager as fm
from dia.learner.advertising import GP_Learner


class Step2:
    def __init__(self, max_bid, max_price, n_bids, n_prices, n_obs, noise_std_n_clicks, noise_std_conv_rate,
                 noise_std_cost_x_click):
        # Bid settings
        self.max_bid = max_bid
        self.max_price = max_price
        self.n_bids = n_bids
        self.n_prices = n_prices
        self.bids = np.linspace(0, self.max_bid, self.n_bids)
        self.prices = np.linspace(0, self.max_price, self.n_prices)

        self.n_obs = n_obs
        self.noise_std_n_clicks = noise_std_n_clicks
        self.noise_std_conv_rate = noise_std_conv_rate
        self.noise_std_cost_x_click = noise_std_cost_x_click

    def generate_observations(self, f_to_estimate, x, customer_class):
        """Function to generate observations. Considering number of clicks and cost per click, the function depends on
        the bid (x == bid). Considering the conversion rate function, the function depends on the price (x == price)"""
        if f_to_estimate == "n_clicks":
            # x = bid
            n = fm.get_n_click(x, customer_class)
            return n + np.random.normal(0, self.noise_std_n_clicks, size=n.shape)
        if f_to_estimate == "cost_per_click":
            # x = bid
            return fm.get_cost_x_click(x) + np.random.normal(0, self.noise_std_cost_x_click,
                                                             size=fm.get_cost_x_click(x).shape)
        if f_to_estimate == "conv_rate":
            # x = price
            return fm.conv_rate(x, customer_class) + np.random.normal(0, self.noise_std_conv_rate,
                                                                      size=fm.conv_rate(x, customer_class).shape)

    def estimate_n_clicks(self):
        f = "n_clicks"
        x_obs = np.array([])
        y_obs = np.array([])

        for i in range(0, self.n_obs):
            gp_learner = GP_Learner(self.noise_std_n_clicks)

            new_x_obs = np.random.choice(self.bids, 1)
            # 1 is the customer class
            new_y_obs = self.generate_observations(f, new_x_obs, 1)

            x_obs = np.append(x_obs, new_x_obs.astype(float))
            y_obs = np.append(y_obs, new_y_obs.astype(float))

            X = np.atleast_2d(x_obs).T
            Y = y_obs.ravel()

            gp_learner.learn(X, Y, self.bids)

            plt.figure(i)
            plt.plot(gp_learner.x_pred, fm.get_n_click(gp_learner.x_pred, 1), 'r', label=r'$n clicks(bid)$')
            plt.plot(X.ravel(), Y, 'ro', label=u'observed clicks')
            plt.plot(gp_learner.x_pred, gp_learner.y_pred, 'b-', label=u'predicted clicks')
            plt.fill(np.concatenate([gp_learner.x_pred, gp_learner.x_pred[::-1]]),
                     np.concatenate([gp_learner.y_pred - 1.96 * gp_learner.sigma, (gp_learner.y_pred + 1.96 * gp_learner.sigma)[::-1]]),
                     alpha=.5, fc='b', ec='None', label='95% confidence interval')
            plt.xlabel('$bid$')
            plt.ylabel('$n_clicks(bid)$')
            plt.legend(loc='lower right')
            plt.show()

    def estimate_conv_rate(self):
        f = "conv_rate"
        x_obs = np.array([])
        y_obs = np.array([])

        for i in range(0, self.n_obs):
            gp_learner = GP_Learner(self.noise_std_conv_rate)

            new_x_obs = np.random.choice(self.prices, 1)
            # 1 is the customer class
            new_y_obs = self.generate_observations(f, new_x_obs, 1)

            x_obs = np.append(x_obs, new_x_obs.astype(float))
            y_obs = np.append(y_obs, new_y_obs.astype(float))

            X = np.atleast_2d(x_obs).T
            Y = y_obs.ravel()

            gp_learner.learn(X, Y, self.bids)

            plt.figure(i)
            plt.plot(gp_learner.x_pred, fm.conv_rate(gp_learner.x_pred, 1), 'r', label=r'$rate(price)$')
            plt.plot(X.ravel(), Y, 'ro', label=u'observed rate')
            plt.plot(gp_learner.x_pred, gp_learner.y_pred, 'b-', label=u'predicted rate')
            plt.fill(np.concatenate([gp_learner.x_pred, gp_learner.x_pred[::-1]]),
                     np.concatenate([gp_learner.y_pred - 1.96 * gp_learner.sigma, (gp_learner.y_pred + 1.96 * gp_learner.sigma)[::-1]]),
                     alpha=.5, fc='b', ec='None', label='95% confidence interval')
            plt.xlabel('$price$')
            plt.ylabel('$rate(price)$')
            plt.legend(loc='lower right')
            plt.show()

    def estimate_cost_x_click(self):
        f = "cost_per_click"
        x_obs = np.array([])
        y_obs = np.array([])

        for i in range(0, self.n_obs):
            gp_learner = GP_Learner(self.noise_std_cost_x_click)

            new_x_obs = np.random.choice(self.prices, 1)
            new_y_obs = self.generate_observations(f, new_x_obs, 0)

            x_obs = np.append(x_obs, new_x_obs.astype(float))
            y_obs = np.append(y_obs, new_y_obs.astype(float))

            X = np.atleast_2d(x_obs).T
            Y = y_obs.ravel()

            gp_learner.learn(X, Y, self.bids)

            plt.figure(i)
            plt.plot(gp_learner.x_pred, fm.get_cost_x_click(gp_learner.x_pred), 'r', label=r'$cost(bid)$')
            plt.plot(X.ravel(), Y, 'ro', label=u'observed cost')
            plt.plot(gp_learner.x_pred, gp_learner.y_pred, 'b-', label=u'predicted cost')
            plt.fill(np.concatenate([gp_learner.x_pred, gp_learner.x_pred[::-1]]),
                     np.concatenate([gp_learner.y_pred - 1.96 * gp_learner.sigma, (gp_learner.y_pred + 1.96 * gp_learner.sigma)[::-1]]),
                     alpha=.5, fc='b', ec='None', label='95% confidence interval')
            plt.xlabel('bid')
            plt.ylabel('$cost(bid)$')
            plt.legend(loc='lower right')
            plt.show()

