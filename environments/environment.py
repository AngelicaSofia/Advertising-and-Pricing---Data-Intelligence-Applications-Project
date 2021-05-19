from dia.environments.advertising_environment import *
from dia.environments.pricing_environment import *


class Scenario(object):
    """
    It represents a scenario on which the experiments is tested (simulation of the reality with a sort of model)

    Assumptions:
     - the number of sub campaigns remains constant
     - the number of features can be different than the number of classes (==number of subcampaigns)
     - the time horizon of the pricing and advertisement has to be the same
     - the sum of user distribution of all type of users (i.e. users with all features specified) is 1
    """

    def __init__(self, n_sub_campaigns, n_user_features, poisson1, poisson2, poisson3):

        self.n_sub_campaigns = n_sub_campaigns
        self.n_user_features = n_user_features
        self.accepted_selling_price_class1 = 0
        self.accepted_selling_price_class2 = 0
        self.accepted_selling_price_class3 = 0
        """Assuming that the inter arrival time between visits is exponential, then we can imagine the distribution of
        the visits as a Poisson. Number of visits is one month can be approximated as the value of lambda, that is the
        parameter of Poisson distribution"""
        self.poisson1 = poisson1  # distribution of the visits to the website for class 1
        self.poisson2 = poisson2  # distribution of the visits to the website for class 2
        self.poisson3 = poisson3  # distribution of the visits to the website for class 3
        self.advertising_campaign = None
        self.pricing_environment = None
        self.bidding_environment = None #Step 5

    def get_n_sub_campaigns(self):
        return self.n_sub_campaigns

    def get_n_user_features(self):
        return self.n_user_features

    def set_price1(self, price):
        self.accepted_selling_price_class1 = price

    def set_price2(self, price):
        self.accepted_selling_price_class2 = price

    def set_price3(self, price):
        self.accepted_selling_price_class3 = price

    def set_advertising_environment(self, bids, weights):
        self.advertising_campaign = Campaign(bids, weights)

    def set_pricing_environment(self, n_arms, probabilities):
        self.pricing_environment = pricing_environment(n_arms, probabilities)

    def set_bidding_environment(self, n_arms, probabilities):
        self.bidding_environment = pricing_environment(n_arms, probabilities)

