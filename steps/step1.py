import pandas as pd
import numpy as np

from dia.environments import function_manager as fm
from dia.utils import logger


class Step1:
    def __init__(self):
        self.best_price = 0
        self.best_bid = 0
        self.max_obj = 0

    def joint_bidding_pricing_enumeration(self, scenario, max_price, max_bid, path_obj, path_delta):
        """Assuming that the average of the Poisson distribution for the number of future visits of customers of class C
        is used, we directly take the value of lambda for each class to count the future visits in average"""
        scenario.poisson1 = scenario.poisson1 / 30
        scenario.poisson2 = scenario.poisson2 / 30
        scenario.poisson3 = scenario.poisson3 / 30
        """Value of lambda is divided by 30 as we are considering the daily value in 30 days"""
        obj_fun_table = logger.create_table(max_price, max_bid)
        delta_obj_fun_table = logger.create_table(max_price, max_bid)
        for price in range(0, max_price):
            prec_obj = 0
            for b in range(0, max_bid):
                part1 = (price *
                         fm.conv_rate(price, 1) *
                         fm.get_n_click(b, 1) *
                         (1 + scenario.poisson1)) - (fm.get_cost_x_click(b) *
                                                     fm.get_n_click(b, 1))
                part2 = (price *
                         fm.conv_rate(price, 2) *
                         fm.get_n_click(b, 2) *
                         (1 + scenario.poisson2)) - (fm.get_cost_x_click(b) *
                                                     fm.get_n_click(b, 2))
                part3 = (price *
                         fm.conv_rate(price, 3) *
                         fm.get_n_click(b, 3) *
                         (1 + scenario.poisson3)) - (fm.get_cost_x_click(b) *
                                                     fm.get_n_click(b, 3))
                obj_fun = part1 + part2 + part3
                delta = obj_fun - prec_obj

                prec_obj = obj_fun
                logger.update_table(obj_fun_table, price, b, obj_fun)
                logger.update_table(delta_obj_fun_table, price, b, delta)
                """The algorithm stops the first time the delta between the objective function and the previous one is not
                greater than 10% the value of the previous objective function. This means that the objective function stops
                improving even if we increase the bid values."""
                if delta < (obj_fun / 100) and b > 0 and price > 0:
                    logger.table_to_csv(obj_fun_table, path_obj)
                    logger.table_to_csv(delta_obj_fun_table, path_delta)
                    break

        self.best_bid_price()
        print("Obtained values for the joint bidding/pricing strategy with enumeration: ")
        print("Best Price: " + str(self.best_price))
        print("Best Bid: " + str(self.best_bid))
        print("Objective Function: " + str(self.max_obj))
        print("\n")

    def best_bid_price(self):
        df = pd.read_csv('results/step1/obj_fun.csv')
        self.max_obj = df.to_numpy().max()
        coordinates = [(x, df.columns[y]) for x, y in zip(*np.where(df.values == self.max_obj))]
        best = coordinates[0]
        self.best_price = best[0]
        self.best_bid = int(best[1])

