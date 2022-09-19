# ________imports___________
import numpy as np
from scipy.stats import multinomial
import random
from tqdm import tqdm
from pathos.multiprocessing import ProcessingPool as Pool

# ________constants___________
MAX_WAREHOUSE = 11
MAX_IN_CIRC = 11
MAX_PRICE = 10
CUSTOMER_ARRIVING = 20
PROD_PRICE = 3
ɣ = 0.99

# ________initialization___________

V = np.zeros((MAX_IN_CIRC, MAX_WAREHOUSE))

# each value in policy array should equal to one action to take --> an action is a triple
π = np.zeros((MAX_IN_CIRC, MAX_WAREHOUSE), dtype=object)
for (x, y), _ in np.ndenumerate(π):
    π[x][y] = [random.randint(1, 9), random.randint(1, 9), random.randint(1, 9)]


# ________helper function___________

def softmax(preferences: np.array) -> np.array:
    preferences = np.minimum(np.ones(len(preferences)) * 20,
                             preferences)  # This avoids an overflow error in the next line
    exp_preferences = np.exp(preferences)
    return exp_preferences / sum(exp_preferences)


def generate_purchase_probabilities_from_offer(price) -> np.array:
    new_buy_price, _, rebuy_price = price

    nothingpreference = 1
    preferences = []

    rebuy_price = rebuy_price + 1
    price_new = new_buy_price + 1

    assert rebuy_price >= 1 and price_new >= 1, 'prices to be >= 1'

    ratio_old = 5.5 / rebuy_price - np.exp(rebuy_price - 5)
    ratio_new = 10 / price_new - np.exp(price_new - 8)
    preferences += [ratio_new, ratio_old, nothingpreference]

    return softmax(np.array(preferences))


def generate_return_probabilities_from_offer(price) -> np.array:
    """
    This method tries a more sophisticated version of generating return probabilities.
    The owner prefers higher rebuy prices.
    If the rebuy price is very low, the owner will just throw away his product more often. Holding the product is the fallback option.
    Check the docstring in the superclass for interface description.
    """

    new_buy_price, used_buy_price, rebuy_price = price

    holding_preference = 1
    return_preferences = []

    price_refurbished = used_buy_price + 1
    price_new = new_buy_price + 1
    price_rebuy = rebuy_price + 1
    # only one vendor ...
    best_rebuy_price = price_rebuy
    lowest_purchase_offer = min(price_refurbished, price_new)
    return_preferences.append(price_rebuy)

    discard_preference = lowest_purchase_offer - best_rebuy_price

    return softmax(np.array([holding_preference, discard_preference] + return_preferences))


def how_to_get_from_circ_to_circ(current_state, next_state, price):
    current_circ, current_warehouse = current_state
    OWNER_ARRIVING = int(current_circ * 0.05)
    next_circ, next_warehouse = next_state

    new_buy_price, used_buy_price, rebuy_price = price
    pr_newbuy, pr_usedbuy, pr_nothing = generate_purchase_probabilities_from_offer(price)
    pr_hold, pr_throwaway, pr_sellback = generate_return_probabilities_from_offer(price)

    pmf_customer = multinomial(CUSTOMER_ARRIVING, [pr_newbuy, pr_usedbuy, pr_nothing])
    pmf_owner = multinomial(OWNER_ARRIVING, [pr_hold, pr_throwaway, pr_sellback])

    ret_value = []
    # customers...
    for new_buy in range(0, CUSTOMER_ARRIVING + 1):
        for used_buy in range(0, CUSTOMER_ARRIVING + 1):
            # owners...
            if new_buy + used_buy <= CUSTOMER_ARRIVING:
                for throw_away in range(0, OWNER_ARRIVING + 1):
                    for sell_back in range(0, OWNER_ARRIVING + 1):
                        if throw_away + sell_back <= OWNER_ARRIVING:

                            # regarding the max expression ... we can go from (x, 5) -> (x, 0) by 5 used sells, but also with 6,...,20
                            if current_circ + new_buy - throw_away - sell_back == next_circ and max(0,
                                                                                                    current_warehouse - used_buy) + sell_back == next_warehouse:
                                oversell = max(used_buy - current_warehouse, 0)
                                prob_cust = pmf_customer.pmf(
                                    [new_buy, used_buy, CUSTOMER_ARRIVING - new_buy - used_buy])
                                prob_owner = 1 if OWNER_ARRIVING == 0 else pmf_owner.pmf(
                                    [OWNER_ARRIVING - throw_away - sell_back, throw_away, sell_back])

                                prob = prob_owner * prob_cust
                                reward = calculate_profit(new_buy, used_buy, sell_back, oversell, price)

                                # print(
                                #     f'\t{new_buy} neu verkauf, {throw_away} wegwurf, {sell_back} zurueckkauf, {used_buy} gebraucht verkauf, {oversell} overselling')
                                # print(f"prob: {prob}")
                                # print(f"reward: {reward}")
                                ret_value.append([reward, prob])

    return ret_value


def calculate_profit(new_buy, used_buy, sell_back, oversell, price):
    new_buy_price, used_buy_price, rebuy_price = price
    reward = 0

    reward += new_buy * (new_buy_price - PROD_PRICE)
    possible_sales = used_buy - oversell
    reward += possible_sales * used_buy_price
    reward -= oversell * 2 * MAX_PRICE

    # owner
    reward -= rebuy_price * sell_back
    return reward


def get_state_value(args):
    current_state, action, value_matrix = args
    current_circ, current_warehouse = current_state

    lower_circ_limit = current_circ + (-1 * int(current_circ * 0.05))
    upper_circ_limit = min(current_circ + CUSTOMER_ARRIVING, MAX_IN_CIRC - 1)

    lower_warehouse_limit = min(current_warehouse - 20, 0)
    upper_warehouse_limit = min(current_warehouse + 20, MAX_WAREHOUSE - 1)

    values = []

    for circ in range(lower_circ_limit, upper_circ_limit + 1):
        for warehouse in range(lower_warehouse_limit, upper_warehouse_limit + 1):
            results = how_to_get_from_circ_to_circ(current_state, (circ, warehouse), action)
            if len(results) > 0:
                # results = [ [reward, prop], ... ]
                value = sum(map(lambda entry: (entry[0] + value_matrix[circ][warehouse] * ɣ) * entry[1], results))
                values.append(value)

    result = sum(values)
    return result


# ________policy evaluation___________

def policy_evaluation():
    pool = Pool(16)
    while True:
        Δ = 0
        for circ in tqdm(range(V.shape[0])):
            prev_values = np.copy(V[circ])
            new_values = pool.map(get_state_value,
                                  [((circ, refurb_warehouse), list(π[circ][refurb_warehouse]), np.copy(V)) for
                                   refurb_warehouse in range(V.shape[1])])
            V[circ] = new_values
            for refurb_warehouse in range(V.shape[1]):
                Δ = max(Δ, abs(prev_values[refurb_warehouse] - V[circ][refurb_warehouse]))

        print(Δ)
        if Δ < 1:
            print("policy evaluation completed.")
            # np.save('value.npy', V)
            # np.save('policy.npy', π)
            break


def find_best_action(args):
    (current_circ, current_warehouse), V = args

    arg_max = [(0, 0, 0), -np.inf]
    for price_new in range(0, 11):
        for price_used in range(0, 11):
            for rebuy_price in range(0, 11):
                reward = get_state_value(
                    [(current_circ, current_warehouse), (price_new, price_used, rebuy_price), np.copy(V)])

                if reward > arg_max[1]:
                    arg_max[0] = (price_new, price_used, rebuy_price)
                    arg_max[1] = reward
    return arg_max[0]


def policy_improvement():
    pool = Pool(16)
    policy_stable = True
    with tqdm(total=V.shape[0]) as pbar:
        for circ in range(V.shape[0]):
            old_actions = np.copy(π[circ])
            new_actions = pool.map(find_best_action,
                                   [((circ, refurb_warehouse), np.copy(V)) for
                                    refurb_warehouse in range(V.shape[1])])
            new_actions = list(map(lambda x: list(x), new_actions))
            pbar.update(1)
            π[circ] = [
                [new_actions[refurb_warehouse][0], new_actions[refurb_warehouse][1], new_actions[refurb_warehouse][2]]
                for refurb_warehouse in range(V.shape[1])]
            for refurb_warehouse in range(V.shape[1]):
                if old_actions[refurb_warehouse] != new_actions[refurb_warehouse]:
                    policy_stable = False
    if policy_stable:
        print("found a stable policy!")
        np.save('value_stable.npy', V)
        np.save('policy_stable.npy', π)

    return policy_stable


if __name__ == '__main__':
    # mp.freeze_support()
    V = np.load("value.npy")
    π = np.load("policy.npy", allow_pickle=True)

    while True:
        policy_evaluation()
        stable = policy_improvement()
        if stable:
            print("we are done here...")
            break
