import pandas as pd
import numpy as np


def process_raw_data(df):
    """
    Function to process the raw data by converting the 'Data' column to datetime and sorting the dataframe.
    Args:
        df (DataFrame): The dataframe to be processed.
    Returns:
        DataFrame: Returns the processed dataframe.
    """

    df['Data'] = pd.to_datetime(df['Data'], format='%d/%m/%Y')
    df = df.sort_values(by='Data', ascending=True)
    return df


def get_consistent_data(df):
    """
    Function to return data with a consistency level of 2, resampled by the start of each month.
    Args:
        df (DataFrame): The dataframe to be filtered.
    Returns:
        DataFrame: Returns the filtered and resampled dataframe.
    """

    consistent_data = df.loc[df['NivelConsistencia']
                             == 2, ["Data", "Maxima"]].copy()

    if consistent_data.shape[0] < 10:
        consistent_data.drop(consistent_data.index, inplace=True)
        return consistent_data

    consistent_data = consistent_data.set_index('Data')
    consistent_data = consistent_data.resample('MS')
    consistent_data = consistent_data.ffill()
    consistent_data = consistent_data.reset_index()

    return consistent_data


def get_raw_data(df):
    """
    Function to return raw data with a consistency level of 1, resampled by the start of each month.
    Args:
        df (DataFrame): The dataframe to be filtered.
    Returns:
        DataFrame: Returns the filtered and resampled dataframe.
    """

    raw_data = df.loc[df['NivelConsistencia'] == 1, [
        "Data", "Maxima"]].reset_index(drop=True)

    raw_data = raw_data.set_index('Data')
    raw_data = raw_data.resample('MS')
    raw_data = raw_data.ffill()
    raw_data = raw_data.reset_index()

    return raw_data


def merge_and_fill_data(consistent_data, raw_data):
    """
    Function to merge the consistent and raw data, and fill in missing values.
    Args:
        consistent_data (DataFrame): The consistent data dataframe.
        raw_data (DataFrame): The raw data dataframe.
    Returns:
        DataFrame: Returns the merged dataframe with filled missing values.
    """

    # Find last date in both DataFrames
    last_date = min(consistent_data['Data'].max(), raw_data['Data'].max())

    # Filter DataFrames until last date
    consistent_data = consistent_data.loc[consistent_data['Data'] <= last_date]
    raw_data = raw_data.loc[raw_data['Data'] <= last_date]

    merged_df = pd.merge(consistent_data, raw_data,
                         on="Data", how="left", suffixes=('', '_y'))

    condition = (merged_df['Maxima'].isna()) | (merged_df['Maxima'] == 0)
    merged_df.loc[condition, 'Maxima'] = merged_df.loc[condition, 'Maxima_y']

    # merged_df['Maxima'].fillna(merged_df['Maxima_y'], inplace=True)
    merged_df.drop('Maxima_y', axis=1, inplace=True)

    merged_df['Maxima'].fillna(0, inplace=True)
    merged_df.loc[merged_df['Maxima'] == 0, 'Maxima'] = 0

    return merged_df


def remove_out_of_cycle_data(df):
    """
    Function to remove data that falls outside of the hydrological cycle (October to September).
    Args:
        df (DataFrame): The dataframe to be filtered.
    Returns:
        DataFrame: Returns the filtered dataframe.
    """

    index_first_sep = df.loc[df['Data'].dt.month == 9].index[0]
    index_last_oct = df.loc[df['Data'].dt.month == 10].index[-1]
    index_last = df.shape[0] - 1

    inicial_drop_range = df.iloc[0:index_first_sep + 1]
    final_drop_range = df.iloc[index_last_oct:index_last + 1]

    df = df.drop(inicial_drop_range.index)
    df = df.drop(final_drop_range.index)

    df = df.reset_index(drop=True)

    return df


def add_hydrological_year(df):
    """
    Function to add a new column for the hydrological year and group the data by the maxima per hydrological year.
    Args:
        df (DataFrame): The dataframe to be processed.
    Returns:
        DataFrame: Returns the dataframe with the new 'AnoHidrologico' and 'ln_Pmax_anual' columns.
    """

    df["AnoHidrologico"] = df["Data"].dt.year.where(
        df["Data"].dt.month >= 10, df["Data"].dt.year - 1)

    hydrological_year_df = df.groupby("AnoHidrologico")["Maxima"].max(
    ).reset_index().rename(columns={"Maxima": "Pmax_anual"})

    hydrological_year_df['ln_Pmax_anual'] = np.log(
        hydrological_year_df['Pmax_anual'])

    hydrological_year_df = hydrological_year_df.sort_values(
        by='Pmax_anual', ascending=False).reset_index(drop=True)

    return hydrological_year_df


def main(raw_df):
    """
    Main function to process the raw data, get consistent and raw data, merge and fill the data, 
    remove out of cycle data, and add a hydrological year.
    Args:
        raw_df (DataFrame): The raw dataframe to be processed.
    Returns:
        DataFrame: Returns the processed dataframe grouped by the maxima per hydrological year.
    """

    raw_df = raw_df.fillna(0)
    rain_data = process_raw_data(raw_df)

    consistent_rain_data = get_consistent_data(rain_data)
    raw_rain_data = get_raw_data(rain_data)

    if consistent_rain_data.empty:
        print("consistent_rain_data is empty")
        consistent_rain_data = raw_rain_data

    filled_rain_data = merge_and_fill_data(consistent_rain_data, raw_rain_data)

    filled_rain_data = remove_out_of_cycle_data(filled_rain_data)

    hydrological_year_data = add_hydrological_year(filled_rain_data)

    if hydrological_year_data.shape[0] < 10:
        hydrological_year_data.drop(hydrological_year_data.index, inplace=True)
        return hydrological_year_data

    # hydrological_year_data.to_csv('hydrological_year_data.csv', sep=',')

    return hydrological_year_data
