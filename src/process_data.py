import pandas as pd
import numpy as np

# Constants
DATE_FORMAT = '%d/%m/%Y'


def process_raw_data(df):
    """
    Function to process the raw data by converting the 'Data' column to datetime and sorting the dataframe.
    """
    df['Data'] = pd.to_datetime(df['Data'], format=DATE_FORMAT)
    df = df.sort_values(by='Data', ascending=True)
    return df


def resample_data(df):
    """
    Function to resample data by the start of each month.
    """
    df = df.set_index('Data')
    df = df.resample('MS').first()
    df = df.fillna(0)
    df = df.reset_index()
    return df


def get_consistent_data(df):
    """
    Function to return data with a consistency level of 2, resampled by the start of each month.
    """
    consistent_data = df.loc[df['NivelConsistencia']
                             == 2, ["Data", "Maxima"]].copy()
    if consistent_data.empty:
        return pd.DataFrame()
    consistent_data = resample_data(consistent_data)
    return consistent_data


def get_raw_data(df):
    """
    Function to return raw data with a consistency level of 1, resampled by the start of each month.
    """
    raw_data = df.loc[df['NivelConsistencia'] == 1,
                      ["Data", "Maxima"]].reset_index(drop=True)
    raw_data = resample_data(raw_data)
    return raw_data


def merge_and_fill_data(consistent_data, raw_data):
    """
    Function to merge the consistent and raw data, and fill in missing values.
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
    merged_df.drop('Maxima_y', axis=1, inplace=True)
    merged_df['Maxima'].fillna(0, inplace=True)
    merged_df.loc[merged_df['Maxima'] == 0, 'Maxima'] = 0
    return merged_df


def remove_out_of_cycle_data(df):
    """
    Function to remove data that falls outside of the hydrological cycle (October to September).
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


def add_water_year(df):
    """
    Function to add a new column for the hydrological year and group the data by the maxima per hydrological year.
    Args:
        df (DataFrame): The dataframe to be processed.
    Returns:
        DataFrame: Returns the dataframe with the new 'AnoHidrologico' and 'ln_Pmax_anual' columns.
    """

    df["AnoHidrologico"] = df["Data"].dt.year.where(
        df["Data"].dt.month >= 10, df["Data"].dt.year - 1)

    water_year_df = df.groupby("AnoHidrologico")["Maxima"].max(
    ).reset_index().rename(columns={"Maxima": "Pmax_anual"})

    # Loop to remove initial values of 'Pmax_anual' in dataframe equals 0
    while not water_year_df.empty and water_year_df.iloc[0]['Pmax_anual'] == 0:
        water_year_df = water_year_df.iloc[1:].reset_index(drop=True)

    # Loop to remove las values of 'Pmax_anual' in dataframe equals 0
    while not water_year_df.empty and water_year_df.iloc[-1]['Pmax_anual'] == 0:
        water_year_df = water_year_df.iloc[:-1].reset_index(drop=True)


    water_year_df['ln_Pmax_anual'] = np.log(
        water_year_df['Pmax_anual'])

    water_year_df = water_year_df.sort_values(
        by='Pmax_anual', ascending=False).reset_index(drop=True)

    return water_year_df


def check_empty_year(water_year_data):
    """Check for years with empty 'Pmax_anual' and return the cleaned water_year_data without those years."""

    empty_years = water_year_data[water_year_data['Pmax_anual']
                                  == 0]['AnoHidrologico'].tolist()
    empty_years.sort()

    water_year_data = water_year_data[water_year_data['Pmax_anual'] != 0]

    if len(empty_years) == 0:
        empty_years = False
    return water_year_data, empty_years


def main(raw_df):
    """
    Main function to process the raw data, get consistent and raw data,
    merge and fill the data, remove out of cycle data, and add a water year.
    """

    raw_df = raw_df.fillna(0)
    rain_data = process_raw_data(raw_df)
    consistent_rain_data = get_consistent_data(rain_data)
    raw_rain_data = get_raw_data(rain_data)

    if consistent_rain_data.empty:
        empty_consistent_data = True
        filled_rain_data = raw_rain_data
    else:
        empty_consistent_data = False
        filled_rain_data = merge_and_fill_data(
            consistent_rain_data, raw_rain_data)

    filled_rain_data = remove_out_of_cycle_data(filled_rain_data)

    water_year_data = add_water_year(filled_rain_data)

    water_year_data, empty_years = check_empty_year(water_year_data)

    year_range = None

    if water_year_data.shape[0] < 10:
        water_year_data.drop(water_year_data.index, inplace=True)
        return water_year_data, empty_consistent_data, year_range, empty_years

    year_range = {
        "first_year": str(water_year_data['AnoHidrologico'].min()),
        "last_year": str(water_year_data['AnoHidrologico'].max())
    }

    return water_year_data, empty_consistent_data, year_range, empty_years
