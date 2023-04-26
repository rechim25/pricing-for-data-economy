import itertools
import pickle
import numpy as np
from typing import Union
from tqdm import tqdm


# Prediction gain functions as defined by buyer
## Linear prediction gain
def linear_G(delta: Union[float, np.array]):
    return np.maximum(0, delta)


## Diminishing return prediction gain function
def diminishing_returns_G(delta: Union[int, float, np.array]):
    return np.log(np.maximum(1, delta))


## Linear prediction gain capped within interval
def linear_capped_G(delta: Union[int, float, np.array], lo_delta: float, hi_delta: float):
    return np.minimum(hi_delta, np.maximum(0, delta - lo_delta))


def normalize(x: np.array):
    return (x - np.min(x)) / np.ptp(x)


def expected_percentage_profits_MC(
        beta_params_range: Union[list, np.array], 
        n_repetitions: int = 10000, 
) -> dict:
    """
    Perform Monte Carlo experiment to obtain the average profits for UR and nonUR strategies.
    Thsi experiment is price scale-agnostic, meaning we sample the prediction gain (and thus profit)
    relative to the price of the dataset. We restrict the possible profit to be in between 0-100%
    of the price. 
    Assumptions:
        - Prediction gain and profit are restricted to -100% to 100% of the dataset price.
        - The buyer's unit value per 1 unit of prediction gain is 1. Otherwise, it just scales the profit,
        thus we would get similar results but scaled.
        - Buyer has perfect information about the distribution of prediction gain outcomes.
        This is the best case for the strategy without UR and puts it into advantage. If we 
        can prove that UR is better in this case, then UR is better the nonUR overall.
    """
    beta_params = list(itertools.product(beta_params_range, beta_params_range))
    n_total_iter = len(beta_params)
    print('Total number of iterations w.r.t. a, b parameters: %d' % n_total_iter)

    params_to_profits = {}
    for a, b in tqdm(beta_params):
        rng = np.random.default_rng()
        # Compute the buyer-estimated delta performance (mean of beta distribution)
        profit_perc_estimated = (a / (a + b))
        # Sample the true profit as a percentange of price
        profit_percs_true = rng.beta(a, b, size=n_repetitions)

        # Append delta_estimate to deltas_true for prediction gain computation and normalization
        profit_percs = np.zeros(profit_percs_true.shape[0] + 1)
        profit_percs[:-1] = profit_percs_true
        profit_percs[-1] = profit_perc_estimated

        # Normalize true and estimated profits in range [-100, 100] (assumption)
        profit_percs_normalized = normalize(profit_percs) * 200 - 100

        profit_percs_true = profit_percs_normalized[:-1]

        profit_perc_estimated = profit_percs_normalized[-1]

        # Save profit for decision with no Uncertainty Reduction (no UR)
        no_ur_profit_avg = 0
        no_ur_profit_std = 0
        if profit_perc_estimated > 0:
            no_ur_profit_avg = np.mean(profit_percs_true)
            no_ur_profit_std = np.std(profit_percs_true)

        # Save profit for decision with Uncertainty Reduction (with UR)
        ur_profits = np.zeros(n_repetitions)
        idxs = np.where(profit_percs_true > 0) # Indices where rational buyer took decision to purchase
        ur_profits[idxs] = profit_percs_true[idxs] # Do not substract price for UR (x) yet

        # Save results to dict
        params_to_profits[(a, b)] = {
            'no_ur_profit_avg': no_ur_profit_avg,
            'no_ur_profit_std': no_ur_profit_std,
            'ur_profit_avg': np.mean(ur_profits),
            'ur_profit_std': np.std(ur_profits),
            'ur_num_purchases': (ur_profits > 0).sum(),
            'n_rep': n_repetitions
        }
    return params_to_profits



if __name__ == '__main__':
    beta_params_range = np.arange(0.01, 20.01, 0.01)

    # Sample profits with linear prediction gain function
    params_to_profits = expected_percentage_profits_MC(beta_params_range, n_repetitions=10000)
    with open('params_to_profit_percs_lin_4mil.pkl', 'wb') as f:
        pickle.dump(params_to_profits, f)