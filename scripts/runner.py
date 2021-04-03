from argparse import ArgumentParser
from dia.environments import function_manager as fm
from dia.environments import Scenario

import json


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--scenario_config', type=str)
    parser.add_argument('--step1_config', type=str)
    args = parser.parse_args()

    with open(args.scenario_config, 'r') as file:
        scenario_config = json.load(file)
    with open(args.step1_config, 'r') as file:
        step1_config = json.load(file)

    n_sub_campaigns = scenario_config["n_sub_campaigns"]
    n_features = scenario_config["n_features"]
    """scenario.poisson1 = 2
    scenario.poisson2 = 1
    scenario.poisson3 = 0.1"""
    scenario = Scenario(scenario_config["n_sub_campaigns"], scenario_config["n_features"],
                        scenario_config["poisson1"], scenario_config["poisson2"], scenario_config["poisson3"])

    print("Step 1: ")
    max_bid = step1_config["max_bid"]
    max_price = step1_config["max_price"]
    obj_fun = 0
    prec_obj = 0
    part1 = 0
    part2 = 0
    part3 = 0
    obtained_bid = 0
    obtained_price = 0
    """Assuming that the average of the Poisson distribution for the number of future visits of customers of class C
    is used, we directly take the value of lambda for each class to count the future visits in average"""
    scenario.poisson1 = scenario.poisson1 / 30
    scenario.poisson2 = scenario.poisson2 / 30
    scenario.poisson3 = scenario.poisson3 / 30
    """Value of lambda is divided by 30 as we are considering the daily value in 30 days"""
    for price in range(0, max_price):
        print("price: " + str(price))
        for b in range(0, max_bid):
            print("bid: " + str(b))
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
            print("prec_obj:" + str(prec_obj))
            print("obj:" + str(obj_fun))
            delta = obj_fun - prec_obj
            print("delta:" + str(delta))

            prec_obj = obj_fun

            if delta < (obj_fun/10) and b > 0 and price > 0:
                obtained_bid = b
                obtained_price = price
                break

    print("Obtained values for the joint bidding/pricing strategy with enumeration: ")
    print("Price: " + str(obtained_price))
    print("Bid: " + str(obtained_bid))

