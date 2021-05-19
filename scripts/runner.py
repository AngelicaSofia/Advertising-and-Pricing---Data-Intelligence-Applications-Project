from argparse import ArgumentParser
from dia.environments import Scenario
from dia.steps import Step1, Step2, Step3, Step4, Step5

import json
import numpy as np


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--scenario_config', type=str)
    parser.add_argument('--step1_config', type=str)
    parser.add_argument('--step2_config', type=str)
    parser.add_argument('--step3_config', type=str)
    parser.add_argument('--step4_config', type=str)
    parser.add_argument('--step5_config', type=str)
    args = parser.parse_args()

    with open(args.scenario_config, 'r') as file:
        scenario_config = json.load(file)
    with open(args.step1_config, 'r') as file:
        step1_config = json.load(file)
    with open(args.step2_config, 'r') as file:
        step2_config = json.load(file)
    with open(args.step3_config, 'r') as file:
        step3_config = json.load(file)
    with open(args.step4_config, 'r') as file:
        step4_config = json.load(file)
    with open(args.step5_config, 'r') as file:
        step5_config = json.load(file)

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

    #step2 = Step2(max_bid, max_price, n_bids, n_prices, n_obs, noise_std_n_clicks, noise_std_conv_rate,
    #              noise_std_cost_x_click)
    #print("Estimate #clicks: \n")
    #step2.estimate_n_clicks()

    #print("Estimate conversion rate: \n")
    #step2.estimate_conv_rate()

    #print("Estimate cost x click: \n")
    #step2.estimate_cost_x_click()

    ###################################################################################################################
    print("Step 3: ")
    n_clicks = step3_config["number_clicks"]
    cost_per_click = step3_config["cost_per_click"]
    lambda_poisson = step3_config["lambda_poisson"]
    arms = np.linspace((max_price - n_prices + 1), max_price, n_prices)
    # Uniform probabilities
    #probabilities = {i:(1/n_prices) for i in range(n_prices)}
    probabilities = step3_config["probabilities"]

    conv_rate = step3_config["conv_rate"]

    scenario.set_pricing_environment(arms, probabilities)
    step3 = Step3(n_clicks, cost_per_click, lambda_poisson, arms, probabilities, conv_rate)
    step3.execute(scenario, step3_config["time_horizon"], step3_config["n_experiment"])
    print("The best arm (price) is: "+str(step3.best_arm))

    ###################################################################################################################
    print("Step 4: ")
    n_persons = 12000
    step4 = Step4(step4_config["probabilities"], step4_config["prices"], step4_config["features"],
                  step4_config["categories"], n_persons)
    step4.execute(scenario, 1, 1)

    ###################################################################################################################
    print("Step 5: ")
    n_obs = step5_config["n_obs"]
    scenario.set_bidding_environment(step5_config["bids"], step5_config["probabilities"])
    step5 = Step5(lambda_poisson, step5_config["bids"], step5_config["probabilities"], conv_rate, n_obs,
                  noise_std_n_clicks, noise_std_cost_x_click, step5_config["up_p"])
    step5.execute(scenario, step3_config["time_horizon"], step5_config["n_experiment"])
    print("The best arm (bid) is: "+str(step5.best_arm))

    ###################################################################################################################
    print("Step 6: ")

    print("The end!")