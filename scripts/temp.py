"""
obj_fun = 0
    prec_obj = 0
    part1 = 0
    part2 = 0
    part3 = 0
    obtained_bid = 0
    obtained_price = 0
    Assuming that the average of the Poisson distribution for the number of future visits of customers of class C
    is used, we directly take the value of lambda for each class to count the future visits in average
    scenario.poisson1 = scenario.poisson1 / 30
    scenario.poisson2 = scenario.poisson2 / 30
    scenario.poisson3 = scenario.poisson3 / 30
    Value of lambda is divided by 30 as we are considering the daily value in 30 days
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

            The algorithm stops the first time the delta between the objective function and the previous one is not
            greater than 10% the value of the previous objective function. This means that the objective function stops
            improving even if we increase the bid values.
            if delta < (obj_fun/10) and b > 0 and price > 0:
                obtained_bid = b
                obtained_price = price
                break

    print("Obtained values for the joint bidding/pricing strategy with enumeration: ")
    print("Price: " + str(obtained_price))
    print("Bid: " + str(obtained_bid))
"""