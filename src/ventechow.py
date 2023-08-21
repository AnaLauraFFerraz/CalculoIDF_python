import pandas as pd
import numpy as np
from scipy.optimize import minimize
import pprint

# Constants
INITIAL_GUESS = [500, 0.1, 10, 0.7]
OPTIMIZATION_BOUNDS = [(100, 2000), (0, 3), (0, 100), (0, 10)]


def rain_intensity_calculations(k_coefficient_data, coefficients, params, time_interval, dist_r2):
    """Calculates the initial rainfall intensity values for
    different return periods and time intervals."""
    idf_data = pd.DataFrame()
    idf_data["Tr_years"] = [2, 5, 10, 20, 30, 50, 75, 100]

    if dist_r2["max_dist"] == "r2_log_normal" or dist_r2["max_dist"] == "r2_log_pearson":
        idf_data["1day"] = np.power(10, (params["meanw"] +
                                         k_coefficient_data["k"] * params["stdw"]))
    else:
        idf_data["1day"] = params["mean"] + \
            k_coefficient_data["k"] * params["std_dev"]

    idf_data["24h"] = idf_data["1day"] * coefficients["24h"]

    for interval_name in time_interval.keys():
        if interval_name != "24h":
            idf_data[interval_name] = (
                idf_data["24h"] * coefficients[interval_name]) / time_interval[interval_name]

    idf_data["24h"] = idf_data["24h"] / time_interval["24h"]

    return idf_data


def transform_dataframe(idf_data, time_interval):
    """Transforms the original DataFrame to facilitate the calculation of the relative error."""
    rows = [
        {"Tr (years)": tr, "td (min)": interval_value * 60,
         "i_real": idf_data.loc[idf_data["Tr_years"] == tr, td].values[0]}
        for tr in idf_data["Tr_years"]
        for td, interval_value in time_interval.items()
    ]

    return pd.DataFrame(rows)


def add_condition(df):
    """Adds a column to the DataFrame with the condition based on the time duration."""
    rows = []
    for index, row in df.iterrows():
        td_min = row["td (min)"]
        if td_min == 60:
            new_row = row.copy()
            new_row["condition"] = 1
            rows.append(new_row.to_dict())
            new_row["condition"] = 2
            rows.append(new_row.to_dict())
        elif 5 <= td_min < 60:
            row["condition"] = 1
            rows.append(row.to_dict())
        elif 60 < td_min <= 1440:
            row["condition"] = 2
            rows.append(row.to_dict())
        else:
            row["condition"] = 3
            rows.append(row.to_dict())

    return pd.DataFrame(rows)


def apply_i_calculated(df, parameters_1, parameters_2):
    """Applies the Ven Te Chow equation to calculate the
    estimated rainfall intensity (i_calculated) for each row."""
    df["i_calculated"] = df.apply(
        lambda row: calculate_i(
            row, parameters_1) if row["condition"] == 1 else calculate_i(row, parameters_2),
        axis=1
    )
    return df


def calculate_i(row, parameters):
    """Calculates the estimated rainfall intensity."""
    k, m, c, n = parameters
    result = (k * row["Tr (years)"] ** m) / ((c + row["td (min)"]) ** n)
    result = np.where(np.isfinite(result), result, 0)
    return result


def optimize_parameters(df, condition):
    """Optimizes the parameters of the Ven Te Chow equation for a given condition.
    Parameters:
    df (pandas.DataFrame): The DataFrame containing the rainfall data.
    condition (int): The condition to optimize for. This should be 1 for time durations between 5 and 60 minutes, and 2 for other time durations.
    Returns: A tuple containing the optimized parameters (k, m, c, n)."""
    df_condition = df[df['condition'] == condition]

    result = minimize(
        objective_function,
        INITIAL_GUESS,
        args=(df_condition),
        bounds=OPTIMIZATION_BOUNDS,
        method="L-BFGS-B"
    )
    k_opt, m_opt, c_opt, n_opt = result.x
    return k_opt.round(4), m_opt.round(4), c_opt.round(4), n_opt.round(4)


def objective_function(parameters, df):
    """Defines the objective function for optimization."""
    df_temp = df.copy()
    k, m, c, n = parameters
    df_temp["i_calculated"] = calculate_i(df_temp, parameters)
    df_temp = add_relative_error(df_temp)
    return df_temp["relative_error"].sum()


def recalculate_dataframe(df, parameters_1, parameters_2):
    """Recalculates the DataFrame with the optimal parameters found."""
    df_temp = df.copy()
    df_temp["i_calculated"] = df_temp.apply(
        lambda row: calculate_i(
            row, parameters_1) if row["condition"] == 1 else calculate_i(row, parameters_2),
        axis=1
    )
    df_temp = add_relative_error(df_temp)

    df_interval_1 = df_temp[df_temp["condition"] == 1]
    df_interval_2 = df_temp[df_temp["condition"] == 2]
    mean_relative_error_1 = df_interval_1["relative_error"].mean()
    mean_relative_error_2 = df_interval_2["relative_error"].mean()
    mean_relative_errors = {
        "interval_1": mean_relative_error_1,
        "interval_2": mean_relative_error_2
    }
    return mean_relative_errors, df_temp


def add_relative_error(df):
    """Adds a column to the DataFrame with the relative error."""
    df["relative_error"] = abs(
        (df["i_calculated"] - df["i_real"]) / df["i_real"]) * 100
    return df


def handle_dist_name(dist_r2):
    if dist_r2["max_dist"] == 'r2_log_normal':
        chosen_dist = "log-normal"
    elif dist_r2["max_dist"] == 'r2_pearson':
        chosen_dist = "Pearson tipo III"
    elif dist_r2["max_dist"] == 'r2_log_pearson':
        chosen_dist = "log-Pearson tipo III"
    elif dist_r2["max_dist"] == 'r2_gumbel_theo':
        chosen_dist = "Gumbel Teórica"
    elif dist_r2["max_dist"] == 'r2_gumbel_finite':
        chosen_dist = "Gumbel Finita"
    else:
        raise ValueError(f"Invalid distribution type: {dist_r2['max_dist']}")

    return chosen_dist


def main(distribution_data, k_coefficient_data, disaggregation_data,
          params, time_interval, dist_r2, empty_consistent_data, year_range):
    """Main function to calculate optimal parameters and recalculate the DataFrame."""

    idf_data = rain_intensity_calculations(
        k_coefficient_data, disaggregation_data, params, time_interval, dist_r2)

    transformed_df = transform_dataframe(
        idf_data, time_interval)

    initial_parameters = (1000, 0.1, 10, 1)
    transformed_df = add_condition(transformed_df)

    transformed_df = apply_i_calculated(
        transformed_df, initial_parameters, initial_parameters)

    transformed_df = add_relative_error(transformed_df)

    k_opt1, m_opt1, c_opt1, n_opt1 = optimize_parameters(transformed_df, 1)
    k_opt2, m_opt2, c_opt2, n_opt2 = optimize_parameters(transformed_df, 2)

    mean_relative_errors, transformed_df = recalculate_dataframe(
        transformed_df, (k_opt1, m_opt1, c_opt1, n_opt1), (k_opt2, m_opt2, c_opt2, n_opt2))

    chosen_dist = handle_dist_name(dist_r2)

    output = {
        "graph_data": {
            "F": (100*distribution_data["F"]).tolist(),
            "P_dist": distribution_data["Pmax_anual"].tolist()[::-1],
        },
        "parameters": {
            "parameters_1": {
                "k1": k_opt1,
                "m1": m_opt1,
                "c1": c_opt1,
                "n1": n_opt1
            },
            "parameters_2": {
                "k2": k_opt2,
                "m2": m_opt2,
                "c2": c_opt2,
                "n2": n_opt2
            }
        },
        "mean_relative_errors": mean_relative_errors,
        "sample_size_above_30_years": params['size'] >= 30,
        "empty_consistent_data": empty_consistent_data,
        "year_range": year_range,
        "dist": chosen_dist
    }

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(output)
    # transformed_df.to_csv('transformed_df.csv', sep=',')

    return output
