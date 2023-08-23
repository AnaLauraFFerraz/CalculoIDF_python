import numpy as np
from scipy.stats import norm, stats, gamma


def exceedence_calculation(df, sample_size):
    """Calculate exceedence for the given dataframe."""
    df["F"] = (df.index + 1) / (sample_size + 1)
    df["F"] = df["F"].round(4)
    df["one_minus_F"] = 1 - df["F"]
    df["P_log"] = np.log10(df["Pmax_anual"])
    return df


def params_calculation(df, sigmaN, yn, sample_size):
    """Calculate parameters for the given dataframe."""
    mean = df["Pmax_anual"].mean()
    std_dev = df["Pmax_anual"].std()
    meanw = df["P_log"].mean()
    std_devw = df["P_log"].std()
    g = (sample_size / ((sample_size - 1) * (sample_size - 2))) * \
        np.sum(((df["Pmax_anual"] - mean) / std_dev) ** 3)
    alpha = 4/(g*g)
    gw = (sample_size / ((sample_size - 1) * (sample_size - 2))) * \
        np.sum(((df["P_log"] - meanw) / std_devw) ** 3)
    alphaw = 4/(gw*gw)

    params = {
        "size": sample_size,
        "mean": mean,
        "std_dev": std_dev,
        "g": g,
        "alpha": alpha,
        "meanw": meanw,
        "stdw": std_devw,
        "gw": gw,
        "alphaw": alphaw,
        "sigman": sigmaN,
        "yn": yn
    }

    return params


def yn_sigman_calculation(yn_table, sigman_table, sample_size):
    """Function to get 'sigmaN' and 'YN' values from
    the dataframe based on the sample size."""
    yn = yn_table[sample_size]
    sigman = sigman_table[sample_size]
    return sigman, yn


def dist_log_normal(df, params):
    """Function to calculate r2 for the log-normal distribution."""
    df["KN"] = norm.ppf(1 - df["F"])
    df["WTr"] = params["meanw"] + params["stdw"] * df["KN"]
    df["P_log_normal"] = np.power(10, df['WTr'])

    corr_log_normal, _ = stats.pearsonr(df["Pmax_anual"], df["P_log_normal"])
    r2_log_normal = corr_log_normal ** 2
    r2_log_normal = r2_log_normal.round(4)
    return r2_log_normal


def dist_pearson(df, params):
    """Function to calculate r2 for the Pearson distribution."""
    alpha = params["alpha"]

    df['YTR'] = np.where(params["g"] > 0, gamma.ppf(df["one_minus_F"],
                         alpha, scale=1), gamma.ppf(df["F"], alpha, scale=1))

    df["KP"] = (params["g"]/2) * (df['YTR'] - alpha)
    df["P_pearson"] = params["mean"] + params["std_dev"] * df["KP"]

    corr_pearson, _ = stats.pearsonr(df["Pmax_anual"], df["P_pearson"])
    r2_pearson = corr_pearson ** 2
    r2_pearson = r2_pearson.round(4)
    return r2_pearson


def dist_log_pearson(df, params):
    """Function to calculate r2 for the log-Pearson distribution."""
    alphaw = params["alphaw"]

    df['YTRw'] = np.where(params["gw"] > 0, 
                          gamma.ppf(df["one_minus_F"], alphaw, scale=1), 
                          gamma.ppf(df["F"], alphaw, scale=1))

    df["KL_P"] = (params["gw"]/2)*(df['YTRw']-params["alphaw"])
    df["WTr_LP"] = params["meanw"] + params["stdw"] * df["KL_P"]
    df["P_log_pearson"] = np.power(10, df['WTr_LP'])

    corr_log_pearson, _ = stats.pearsonr(df["Pmax_anual"], df["P_log_pearson"])
    r2_log_pearson = corr_log_pearson ** 2
    r2_log_pearson = r2_log_pearson.round(4)
    return r2_log_pearson


def dist_gumbel_theoretical(df, params):
    """Function to calculate r2 for the theoretical Gumbel distribution."""

    df["y"] = df["one_minus_F"].apply(lambda x: -np.log(-np.log(x)))
    df["KG_T"] = 0.7797 * df["y"] - 0.45
    df["P_gumbel_theoretical"] = params["mean"] + \
        params["std_dev"] * df["KG_T"]

    corr_gumbel, _ = stats.pearsonr(
        df["Pmax_anual"], df["P_gumbel_theoretical"])

    r2_gumbel_theo = corr_gumbel ** 2
    r2_gumbel_theo = r2_gumbel_theo.round(4)
    return r2_gumbel_theo


def dist_gumbel_finite(df, params):
    """Function to calculate r2 for the finite Gumbel distribution."""
    df["KG_F"] = (df["y"] - params["yn"]) / params["sigman"]
    df["P_gumbel_finite"] = params["mean"] + params["std_dev"] * df["KG_F"]

    corr_gumbel_finite, _ = stats.pearsonr(
        df["Pmax_anual"], df["P_gumbel_finite"])

    r2_gumbel_finite = corr_gumbel_finite ** 2
    r2_gumbel_finite = r2_gumbel_finite.round(4)
    return r2_gumbel_finite


def dist_calculations(no_oulier_data, params):
    """Function to perform various distribution calculations."""

    r2_log_normal = dist_log_normal(no_oulier_data, params)

    r2_pearson = dist_pearson(no_oulier_data, params)

    r2_log_pearson = dist_log_pearson(no_oulier_data, params)

    r2_gumbel_theo = dist_gumbel_theoretical(no_oulier_data, params)

    r2_gumbel_finite = dist_gumbel_finite(no_oulier_data, params)

    distributions_r2 = {
        "log_normal": r2_log_normal,
        "pearson": r2_pearson,
        "log_pearson": r2_log_pearson,
        "gumbel_theoretical": r2_gumbel_theo,
        "gumbel_finite": r2_gumbel_finite
    }

    max_dist = max(distributions_r2, key=distributions_r2.get)
    max_r2 = distributions_r2[max_dist]
    # max_dist = "pearson"
    # max_r2 = distributions["pearson"]

    dist_r2 = {"max_dist": max_dist,
               "max_value_r2": max_r2}

    return no_oulier_data, dist_r2


def main(no_oulier_data, yn_table, sigman_table):
    """Main function to perform various calculations."""

    sample_size = len(no_oulier_data)

    distribuitions_df = exceedence_calculation(no_oulier_data, sample_size)

    yn, sigmaN = yn_sigman_calculation(yn_table, sigman_table, sample_size)

    params = params_calculation(distribuitions_df, sigmaN, yn, sample_size)

    distributions_data, dist_r2 = dist_calculations(
        distribuitions_df, params)

    return distributions_data, params, dist_r2
