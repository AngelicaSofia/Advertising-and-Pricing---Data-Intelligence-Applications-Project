import numpy as np


def get_n_click(bid, customer):
    """Function that returns a stochastic number of daily clicks of new users  (i.e., that have never clicked before
    these ads) as a function depending on the bid"""
    alpha = 1
    beta = 1
    if customer == 1:
        alpha = 100  # 300
        beta = 0.8  # 0.02
    elif customer == 2:
        alpha = 120  # 500
        beta = 0.2  # 0.003
    elif customer == 3:
        alpha = 70
        beta = 0.06  # 0.02
    n = -1
    while n < 0:
        param = np.tanh(np.random.normal(loc=bid, scale=0.6 * np.tanh(bid)) * beta)
        n_click = alpha * param
        n = n_click
    return n_click


def get_cost_x_click(bid):
    """Function that returns a stochastic cost per click as a function of the bid"""
    cost_per_click = 0
    while cost_per_click < 0:
        cost_per_click = np.random.normal(loc=np.log(bid+1)/8, scale=0.01*np.tanh(bid), size=None)
    return cost_per_click


# class 1: TT
# class 2: TF
# class 3: FF
def conv_rate(price, customer):
    """Function that is a conversion rate providing the probability that a user will buy the item given a price"""
    alpha = 1
    if customer == 1:
        alpha = - 0.25
    elif customer == 2:
        alpha = - 0.5
    elif customer == 3:
        alpha = - 1.5
    conv = np.exp(alpha*price)
    return conv
