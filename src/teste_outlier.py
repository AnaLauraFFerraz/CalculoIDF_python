import numpy as np


def calculate_statistics(df):
    """
    Function to calculate statistical values such as sample size, mean, and standard deviation.
    Args:
        df (DataFrame): The dataframe to be processed.
    Returns:
        tuple: Returns a tuple containing sample size, mean, ln_mean, std, ln_std.
    """

    sample_size = df.shape[0]

    mean = df['Pmax_anual'].mean()
    ln_mean = df['ln_Pmax_anual'].mean()

    std = df['Pmax_anual'].std()
    ln_std = df['ln_Pmax_anual'].std()

    return sample_size, mean, ln_mean, std, ln_std


def calc_critical_values(gb_test, sample_size, ln_p_mean, ln_p_std):
    """
    Function to calculate critical values (t_crit_10, x_h, x_l) based on Grubbs' Test.
    Args:
        gb_test (DataFrame): The dataframe containing the Grubbs' Test.
        sample_size (int): The size of the sample.
        ln_p_mean (float): The natural logarithm of the mean.
        ln_p_std (float): The natural logarithm of the standard deviation.
    Returns:
        tuple: Returns a tuple containing t_crit_10, x_h, x_l.
    """

    t_crit_10 = gb_test.loc[gb_test['Number of observations']
                            == sample_size, 'Upper 10% Significance Level'].values[0]

    k_n_10 = -3.62201 + 6.28446*(sample_size**0.25) - 2.49835*(
        sample_size**0.5) + 0.491436*(sample_size**0.75) - 0.037911*sample_size

    x_h = np.exp(ln_p_mean + k_n_10 * ln_p_std)
    x_l = np.exp(ln_p_mean - k_n_10 * ln_p_std)

    return t_crit_10, x_h, x_l


def remove_outliers(df, p_mean, p_std, t_crit_10, x_h, x_l):
    """
    Function to remove outliers from the data based on the calculated critical values.

    Args:
        df (DataFrame): The dataframe to be processed.
        p_mean (float): The mean.
        p_std (float): The standard deviation.
        t_crit_10 (float): The critical value for Grubbs' Test.
        x_h (float): The upper critical value for Grubbs' Test.
        x_l (float): The lower critical value for Grubbs' Test.

    Returns:
        DataFrame: Returns the dataframe with outliers removed.
    """

    max_outlier = True

    while max_outlier:
        p_max = df['Pmax_anual'].max()
        t_larger = (p_max - p_mean) / p_std

        if t_larger > t_crit_10 and p_max > x_h:
            max_outlier = True
        else:
            max_outlier = False

        if max_outlier:
            df = df.drop(labels=df[df['Pmax_anual'] == p_max].index)

    min_oulier = True

    while min_oulier:
        p_min = df['Pmax_anual'].min()
        t_smaller = (p_mean - p_min) / p_std

        if t_smaller > t_crit_10 and p_min < x_l:
            min_oulier = True
        else:
            min_oulier = False

        if min_oulier:
            df = df.drop(labels=df[df['Pmax_anual'] == p_min].index)

    return df


def main(processed_data, gb_test):
    """
    Main function to calculate statistics, calculate critical values, and remove outliers.
    Args:
        processed_data (DataFrame): The processed dataframe.
        gb_test (DataFrame): The dataframe containing the Grubbs' Test.
    Returns:
        DataFrame: Returns the processed dataframe with outliers removed.
    """

    sample_size, p_mean, ln_p_mean, p_std, ln_p_std = calculate_statistics(
        processed_data)

    t_crit_10, x_h, x_l = calc_critical_values(
        gb_test, sample_size, ln_p_mean, ln_p_std)

    no_outliers_data = remove_outliers(
        processed_data, p_mean, p_std, t_crit_10, x_h, x_l)

    # no_outliers_data.to_csv('no_outliers_data.csv', sep=',')

    return no_outliers_data
