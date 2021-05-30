#fatto replace di bids con budgets

class Campaign:
    def __init__(self, bids, weights, sigma=0.0):
        self.subcampaigns = []
        self.weights = weights

        self.bids = bids
        self.sigma = sigma

    def add_subcampaign(self, label):
        self.subcampaigns.append(
            Subcampaign(label, self.bids, self.weights)
        )

    # round a specific arm
    def round(self, subcampaign_id, pulled_arm):
        return self.subcampaigns[subcampaign_id].round(pulled_arm)

    # round all arms
    def round_all(self):
        table = []
        for subcampaign in self.subcampaigns:
            table.append(subcampaign.round_all())
        return table


class Subcampaign:
    def __init__(self, label, bids, weights):
        self.label = label
        self.weights = weights
        self.bids = bids

    # round a specific arm
    def round(self, pulled_arm, aggregate):
        # aggregate sample
        if aggregate:
            return self.weights * self.round(pulled_arm)
        # disaggregate sample
        else:
            return self.round(pulled_arm)

    # round all arms
    def round_all(self):
        return [self.round(pulled_arm, True) for pulled_arm in range(len(self.bids))]

