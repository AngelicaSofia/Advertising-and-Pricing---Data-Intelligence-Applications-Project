from argparse import ArgumentParser
from dia.environments import Scenario
from dia.steps import Step1

import json


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--scenario_config', type=str)
    parser.add_argument('--step1_config', type=str)
    parser.add_argument('--step2_config', type=str)
    args = parser.parse_args()

    with open(args.scenario_config, 'r') as file:
        scenario_config = json.load(file)
    with open(args.step1_config, 'r') as file:
        step1_config = json.load(file)
    with open(args.step2_config, 'r') as file:
        step2_config = json.load(file)

    n_sub_campaigns = scenario_config["n_sub_campaigns"]
    n_features = scenario_config["n_features"]
    scenario = Scenario(scenario_config["n_sub_campaigns"], scenario_config["n_features"],
                        scenario_config["poisson1"], scenario_config["poisson2"], scenario_config["poisson3"])

    ###################################################################################################################
    print("Step 1: ")
    max_bid = step1_config["max_bid"]
    max_price = step1_config["max_price"]
    path1_obj = step1_config["path_obj"]
    path1_delta = step1_config["path_delta"]
    step1 = Step1()
    step1.joint_bidding_pricing_enumeration(scenario, max_price, max_bid, path1_obj, path1_delta)
    ###################################################################################################################
    print("Step 2: ")
    max_bid = step2_config["max_bid"]
    min_bid = step2_config["min_bid"]
    max_price = step2_config["max_price"]
    min_price = step2_config["min_price"]
    #budgets =
