# Advertising and Pricing
*Project for the course of Data Intelligence Applications at Politecnico di Milano Master Level*

###### Scenario. 
Consider the scenario in which advertisement is used to attract users on an ecommerce website and the users, after the purchase of the first unit of a consumable item, will buy additional units of the same item in future. The goal is to find the best joint bidding and pricing strategy taking into account future purchases.

###### Environment. 
Imagine a consumable item (for which we have an infinite number of units) and two binary features. Imagine three classes of customers C1, C2, C3, each corresponding to a subspace of the features’ space. Each customers’ class is characterized by:
- a stochastic number of daily clicks of new users (i.e., that have never clicked before these ads) as a function depending on the bid;
- a daily stochastic cost per click as a function of the bid;
- a conversion rate function providing the probability that a user will buy the item given a price;
- a distribution probability over the number of times the user will come back to the ecommerce website to buy that item by 30 days after the first purchase (and simulate such visits in future).
Every price available is associated with a margin obtained by the sale that is known beforehand. Do not need to simulate the functioning of the auctions and the other advertisers.

###### Steps. 
You need to complete the following steps.
1. Formulate the objective function when assuming that, once a user makes a purchase with a price p, then the ecommerce will propose the same price p to future visits of the same user and this user will surely buy the item. The revenue function must take into account the cost per click, while there is no budget constraint. Provide an algorithm to find the best joint bidding/pricing strategy and describe its complexity in the number of values of the bids and prices available (assume here that the values of the parameters are known). In the following Steps, assume that the number of bid values are 10 as well as the number of price values.
2. Consider the online learning version of the above optimization problem when the parameters are not known. Identify the random random variables, potential delays in the feedback, and choose a model for each of them when a round corresponds to a single day. Consider a time horizon of one year.
3. Consider the case in which the bid is fixed and learn in online fashion the best pricing strategy when the algorithm does not discriminate among the customers’ classes (and therefore the algorithm works with aggregate data). Assume that the number of daily clicks and the daily cost per click are known. Adopt both an upper-confidence bound approach and a Thompson-sampling approach and compare their performance.
4. Do the same as Step 3 when instead a context-generation approach is adopted to identify the classes of customers and adopt a potentially different pricing strategy per class. In doing that, evaluate the performance of the pricing strategies in the different classes only at the optimal solution (e.g., if prices that are not optimal for two customers’ classes provide different performance, you do not split the contexts). Let us remark that no discrimination of the customers’ classes is performed at the advertising level. From this Step on, choose one approach between the upper-confidence bound one and the Thompson-sampling one.
5. Consider the case in which the prices are fixed and learn in online fashion the best bidding strategy when the algorithm does not discriminate among the customers’ classes. Assume that the conversion probability is known. However, we need to guarantee some form of safety to avoid the play of arms that provide a negative revenue with a given probability. This can be done by estimating the probability distribution over the revenue for every arm and making an arm eligible only when the probability to have a negative revenue is not larger than a threshold (e.g., 20%). Apply this safety constraint after 10 days to avoid that the feasible set of arms is empty, while in the first 10 days choose the arm to pull with uniform probability. Do not discriminate over the customers’ classes.
6. Consider the general case in which one needs to learn the joint pricing and bidding strategy under the safety constraint introduced in Step 5. Do not discriminate over the customers’ classes both for advertising and pricing.
7. Do the same as Step 6 when instead discriminating over the customers’ classes for both advertising and pricing. In doing that, adopt the context structure already discovered in Step 4.

###### Duties. You are required to:
- Produce the Python code.
- Produce a technical report describing the environment, the algorithms and the plot representing the regret and the reward of every algorithm. Provide also a practical application motivating the scenario.
- Produce a presentation as a summary of the technical report.

###### How to run:
Main in scripts/runner.py
N.B. Some points of the steps are commented, to run the single step it is necessary to uncomment the specific step.
