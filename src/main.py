import os
import json
import pandas as pd
from flask import jsonify
from gcs_utils import download_csv_file, delete_blob

from yn_sigman import yn_sigman
from process_data import main as process_data
from outlier_test import main as outlier_test
from distributions import main as distributions
from k_coefficient import main as k_coefficient
from disaggregation_coef import disaggregation_coef
from ventechow import main as ventechow


def load_data(csv_file_path):
    """Function to load the required data for further analysis."""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        input_data = pd.read_csv(csv_file_path, sep=";", encoding='ISO 8859-1', skiprows=12,
                                 decimal=",", usecols=["NivelConsistencia", "Data", "Maxima"], index_col=False)

        # Check if the necessary columns are present
        required_columns = ["NivelConsistencia", "Data", "Maxima"]
        if not all(column in input_data.columns for column in required_columns):
            print(f"CSV file {csv_file_path} does not have the required columns")
            return None

        return input_data
    except FileNotFoundError:
        print(f"File {csv_file_path} not found")
        return None
    except pd.errors.ParserError:
        print(f"Error parsing CSV file {csv_file_path}")
        return None
    except Exception as e:
        print(f"Unexpected error reading CSV file {csv_file_path}: {e}")
        return None


def main(csv_file_path):
    """Main function to process the data, test for outliers, determine the distribution, 
    calculate the k coefficient, and calculate the Ven Te Chow parameters."""
    raw_df = load_data(csv_file_path)
    if raw_df is None:
        print("Error loading data")
        error_loading_data = "Erro ao carregar o arquivo"
        return json.dumps(error_loading_data)

    processed_data, empty_consistent_data, year_range = process_data(raw_df)
    if processed_data.empty:
        insufficient_data = "Dados não são sufientes para completar a análise"
        return json.dumps(insufficient_data)
    
    no_outlier = outlier_test(processed_data)

    yn_table, sigman_table = yn_sigman()

    distribution_data, params, dist_r2 = distributions(
        no_outlier, yn_table, sigman_table)

    disaggregation_data, time_interval = disaggregation_coef()
    k_coefficient_data = k_coefficient(params, dist_r2)

    output = ventechow(distribution_data, k_coefficient_data,
                       disaggregation_data, params, time_interval, dist_r2,
                         empty_consistent_data, year_range)

    return output


def process_request(request):
    """HTTP Cloud Function.
    Args:
        request (flask.Request): The request object.
        <http://flask.pocoo.org/docs/1.0/api/#flask.Request>
    Returns:
        The response text, or any set of values that can be turned into a
        Response object using `make_response`
        <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>.
    """
    # Parse the request body to get the CSV file URL
    request_json = request.get_json(silent=True)
    request_args = request.args

    if request_json and 'csv_file_url' in request_json:
        csv_file_url = request_json['csv_file_url']
    elif request_args and 'csv_file_url' in request_args:
        csv_file_url = request_args['csv_file_url']
    else:
        return jsonify(error="csv_file_url not provided"), 400

    # Download the CSV file from Firebase Cloud Storage
    csv_file_path = download_csv_file(csv_file_url)

    # Process the data
    result = None
    try:
        result = main(csv_file_path)
    except Exception as e:
        print(f"Error processing data: {e}")
        return jsonify(error="Error processing data"), 500

    if not result:
        return jsonify(error="No result from main function"), 500

    # Delete the CSV file from the local machine
    os.remove(csv_file_path)

    # Delete the CSV file from Firebase Storage
    delete_blob(csv_file_url)

    return jsonify(result)


# if __name__ == "__main__":
#     cv = "CalculoIDF_python/src/csv/chuvas_C_01844000_CV.csv"
#     pl = "CalculoIDF_python/src/csv/chuvas_C_01944009_PL.csv"
#     ma = "CalculoIDF_python/src/csv/chuvas_C_02043032_MA.csv"
#     csv_file_path = cv
#     main(csv_file_path)
