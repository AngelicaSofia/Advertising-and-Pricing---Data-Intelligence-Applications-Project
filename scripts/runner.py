from argparse import ArgumentParser
from dia.environments import Scenario
from dia.steps import Step1, Step2

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
    #max_bid = step1_config["max_bid"]
    #max_price = step1_config["max_price"]
    path1_obj = step1_config["path_obj"]
    path1_delta = step1_config["path_delta"]
    #step1 = Step1()
    #step1.joint_bidding_pricing_enumeration(scenario, max_price, max_bid, path1_obj, path1_delta)
    ###################################################################################################################
    print("Step 2: ")
    max_bid = step2_config["max_bid"]
    max_price = step2_config["max_price"]
    n_bids = step2_config["n_values_bid"]
    n_prices = step2_config["n_values_price"]
    n_obs = step2_config["n_obs"]
    noise_std_n_clicks = step2_config["noise_std_n_clicks"]
    noise_std_conv_rate = step2_config["noise_std_conv_rate"]
    noise_std_cost_x_click = step2_config["noise_std_cost_x_click"]

    step2 = Step2(max_bid, max_price, n_bids, n_prices, n_obs, noise_std_n_clicks, noise_std_conv_rate,
                  noise_std_cost_x_click)
    print("Estimate #clicks: \n")
    step2.estimate_n_clicks()

    print("Estimate conversion rate: \n")
    step2.estimate_conv_rate()

    print("Estimate cost x click: \n")
    step2.estimate_cost_x_click()
    ###################################################################################################################
