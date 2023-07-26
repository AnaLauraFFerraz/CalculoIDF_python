import numpy as np
import pandas as pd
from scipy.stats import norm, gamma


def k_coeficient_calculation():
    """Calculate the k coefficient values based on the annual exceedance probabilities."""
    k_coefficient = pd.DataFrame()
    k_coefficient["Tr_anos"] = [2, 5, 10, 20, 30, 50, 75, 100]
    k_coefficient["exceedance"] = 1 / k_coefficient["Tr_anos"]
    k_coefficient["no_exceedance"] = 1 - k_coefficient["exceedance"]
    return k_coefficient


def k_dist_log_normal_calc(k_coefficient, params):
    """Calculate the k coefficient for a log-normal distribution."""

    k_coefficient["k"] = params["meanw"] + \
        norm.ppf(k_coefficient["no_exceedance"]) * params["stdw"]
    return k_coefficient.round(4)


def k_dist_pearson_calc(k_coefficient, params):
    """Calculate the k coefficient for a Pearson distribution."""

    k_coefficient['YTR'] = np.where(params["alpha"] > 0, gamma.ppf(k_coefficient["no_exceedance"],
                                                                   params["alpha"], scale=1), gamma.ppf(k_coefficient["exceedance"], params["alpha"], scale=1))

    k_coefficient["k"] = (params["g"] / 2) * \
        (k_coefficient["YTR"] - params["alpha"])
    return k_coefficient.round(4)


def k_dist_log_pearson_calc(k_coefficient, params):
    """Calculate the k coefficient for a log-Pearson distribution."""

    k_coefficient['YTRw'] = np.where(params["alphaw"] > 0, gamma.ppf(k_coefficient["no_exceedance"],
                                                                     params["alphaw"], scale=1), gamma.ppf(k_coefficient["exceedance"], params["alphaw"], scale=1))

    k_coefficient["k"] = (params["gw"] / 2) * \
        (k_coefficient["YTRw"] - params["alphaw"])
    return k_coefficient.round(4)


def k_dist_gumbel_theoretical_calc(k_coefficient):
    """Calculate the k coefficient for a theoretical Gumbel distribution."""

    k_coefficient["y"] = -np.log(-np.log(k_coefficient["no_exceedance"]))
    k_coefficient["k"] = 0.7797 * k_coefficient["y"] - 0.45
    return k_coefficient.round(4)


def k_dist_gumbel_finite_calc(k_coefficient, params):
    """Calculate the k coefficient for a finite Gumbel distribution."""

    k_coefficient["y"] = -np.log(-np.log(k_coefficient["no_exceedance"]))
    k_coefficient["k"] = (
        k_coefficient["y"] - params["yn"]) / params["sigman"]
    return k_coefficient.round(4)


def main(params, dist_r2):
    """Calculate the k coefficient values based on the type of distribution."""

    k_coefficient = k_coeficient_calculation()

    if dist_r2["max_dist"] == 'r2_log_normal':
        k = k_dist_log_normal_calc(k_coefficient, params)
    elif dist_r2["max_dist"] == 'r2_pearson':
        k = k_dist_pearson_calc(k_coefficient, params)
    elif dist_r2["max_dist"] == 'r2_log_pearson':
        k = k_dist_log_pearson_calc(k_coefficient, params)
    elif dist_r2["max_dist"] == 'r2_gumbel_theo':
        k = k_dist_gumbel_theoretical_calc(k_coefficient)
    elif dist_r2["max_dist"] == 'r2_gumbel_finite':
        k = k_dist_gumbel_finite_calc(k_coefficient, params)
    else:
        raise ValueError(f"Invalid distribution type: {dist_r2['max_dist']}")
    
    return k
