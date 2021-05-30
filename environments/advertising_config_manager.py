import json
import numpy as np


class Advertising_Config_Manager:
    def __init__(self):
        with open('../config/sub_camp_config.json') as json_file:
            data = json.load(json_file)
        campaign = data["campaigns"]

        # Class settings
        self.feature_labels = list(campaign["subcampaigns"].keys())

        # Experiment settings
        self.sigma = campaign["sigma"]
        self.click_functions = {}
        for feature in campaign["subcampaigns"]:
            self.click_functions[feature] = []
            max_value = feature["max_value"]
            speed = feature["speed"]

            assert (max_value >= 0), "Max value not valid for "+feature+": "+str(max_value)
            assert (0 <= speed <= 1), "Speed value not valid for "+feature+": "+str(speed)

            self.click_functions[feature].append(lambda x, s=speed, m=max_value: self.function(x, s, m))

    def function(self, x, s, m):
        return (1 - np.exp(-s*x)) * m
