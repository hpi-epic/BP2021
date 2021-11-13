from competitor import Competitor
import numpy as np
import math

# Both policies are expected to be deterministic

GAMMA = 1
maxprice = 30
maxquality = 10

def softmax(priorities):
    exp_priorities = np.exp(priorities)
    return exp_priorities / sum(exp_priorities)

def probabilities(offers):
    value_agent = (offers[1] + 20) / offers[0] - math.exp(offers[0] - 25)
    value_compet = (offers[3] + 20) / offers[2] - math.exp(offers[2] - 25)
    buy_nothing = max(0, min(offers[0], offers[2]) - 23) + 1
    return softmax([value_agent, value_compet, buy_nothing])

def expected_profit(offers):
    return probabilities(offers)[0] * (offers[0] - 10) * 10

def transition(agentquality, competitorprice, competitorquality, considerprice, compet):
    profit = expected_profit([considerprice, agentquality, competitorprice, competitorquality])
    competitorprice = compet.give_competitors_price([considerprice, agentquality, competitorprice, competitorquality])
    profit += expected_profit([considerprice, agentquality, competitorprice, competitorquality])
    return profit, int(competitorprice)

def evaluate_dp_table(dptable, idx):
    mysum = 0
    for agentquality in range(maxquality):
        for competitorprice in range(maxprice):
            for competitorquality in range(maxquality):
                mysum += dptable[idx][agentquality][competitorprice][competitorquality]
    return mysum / (maxprice * maxquality * maxquality)

def evaluate_policy():
    compet=Competitor()
    dptable = []
    for i in range(2):
        j_inner = []
        for j in range(maxquality):
            k_inner = []
            for k in range(maxprice):
                l_inner = []
                for l in range(maxquality):
                    l_inner.append(0)
                k_inner.append(l_inner)
            j_inner.append(k_inner)
        dptable.append(j_inner)


    for counter in range(50):
        for agentquality in range(maxquality):
            for competitorprice in range(1, maxprice):
                for competitorquality in range(maxquality):
                    bestprice = 0
                    for considerprice in range(1, maxprice):
                        expected_reward, expected_action = transition(agentquality, competitorprice, competitorquality, considerprice, compet)
                        bestprice = max(bestprice, expected_reward + GAMMA * dptable[(counter + 1) % 2][agentquality][expected_action - 1][competitorquality])
                    dptable[counter % 2][agentquality][competitorprice - 1][competitorquality] = bestprice
        # print("After round ", counter, ": ", evaluate_dp_table(dptable, counter % 2))
    
    return evaluate_dp_table(dptable, 1)
