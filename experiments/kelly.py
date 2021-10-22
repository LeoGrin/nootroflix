import numpy as np


def run_bet(p_success, odd, wealth_before, frac_invested):
    success = np.random.binomial(1, p_success)
    wealth_after = wealth_before * (1 - frac_invested) + success * wealth_before * frac_invested * (odd + 1)
    return wealth_after



if __name__ == """__main__""":
    #kelly
    n_bets = 100
    p_success = 0.6
    odd = 1
    frac_invested = 1
    #frac_invested = (p_success - (1 - p_success) / odd)
    n_experiments = 100000
    final_wealths = []
    for experiment in range(n_experiments):
        wealth = 1
        for bet in range(n_bets):
            wealth = run_bet(p_success, odd, wealth, frac_invested)
            if wealth <= 0.0001:
                #print("GAME OVER at bet {}".format(bet))
                break
        final_wealths.append(wealth)
    print(np.mean(final_wealths))
