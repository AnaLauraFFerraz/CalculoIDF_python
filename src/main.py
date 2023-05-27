import json
# import sys
import os
import pandas as pd
from flask import jsonify
from google.cloud import storage

from process_data import main as process_data
from teste_outlier import main as teste_outlier
from distributions import main as distributions
from k_coefficient import main as k_coefficient
from disaggregation_coef import disaggregation_coef
from ventechow import main as ventechow


def load_data(csv_file_path):
    """Function to load the required data for further analysis."""

    script_dir = os.path.dirname(os.path.abspath(__file__))

    input_data = pd.read_csv(csv_file_path, sep=";", encoding='ISO 8859-1', skiprows=12,
                             decimal=",", usecols=["NivelConsistencia", "Data", "Maxima"], index_col=False)

    gb_test_file_path = os.path.join(script_dir, "csv", "Tabela_Teste_GB.csv")
    gb_test = pd.read_csv(gb_test_file_path, sep=",",
                          encoding='ISO 8859-1', decimal=",", index_col=False)

    yn_sigman_file_path = os.path.join(
        script_dir, "csv", "Tabela_YN_sigmaN.csv")
    table_yn_sigman = pd.read_csv(yn_sigman_file_path, sep=",", encoding='ISO 8859-1',
                                  decimal=",", usecols=["Size", "YN", "sigmaN"], index_col=False)

    return input_data, gb_test, table_yn_sigman


def main(csv_file_path):
    """Main function to process the data, test for outliers, determine the distribution, 
    calculate the k coefficient, and calculate the Ven Te Chow parameters."""

    raw_df, gb_test, table_yn_sigman = load_data(csv_file_path)

    processed_data = process_data(raw_df)

    if processed_data.empty:
        insufficient_data = "Dados não são sufientes para completar a análise"
        with open('output/idf_data.json', 'w', encoding='utf-8') as file:
            json.dump(insufficient_data, file)
        return json.dumps(insufficient_data)

    no_outlier = teste_outlier(processed_data, gb_test)

    distribution_data, params, dist_r2 = distributions(
        no_outlier, table_yn_sigman)
    # distribution_data.to_csv('distribution_data.csv', sep=',')

    disaggregation_data, time_interval = disaggregation_coef()

    k_coefficient_data = k_coefficient(params, dist_r2)

    output = ventechow(distribution_data, k_coefficient_data,
                       disaggregation_data, params, time_interval, dist_r2)

    with open('output/idf_data.json', 'w', encoding='utf-8') as json_file:
        json.dump(output, json_file)

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

    # Download the CSV file from Cloud Storage
    bucket_name, blob_name = parse_gcs_url(csv_file_url)
    csv_file_path = download_blob(bucket_name, blob_name)

    # Process the CSV file
    output = main(csv_file_path)

    # Return the output as a JSON response
    return jsonify(output)


def parse_gcs_url(gcs_url):
    """Parse a Google Cloud Storage URL into (bucket_name, blob_name)"""
    # Remove the 'gs://' prefix
    gcs_url = gcs_url[5:]
    bucket_name, blob_name = gcs_url.split('/', 1)
    return bucket_name, blob_name


def download_blob(bucket_name, blob_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    # Download the blob to a local file
    csv_file_path = "/tmp/" + blob_name
    blob.download_to_filename(csv_file_path)

    return csv_file_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Por favor, forneça o caminho do arquivo CSV como argumento")
    else:
        csv_file_path = sys.argv[1]
        main(csv_file_path)

# if __name__ == "__main__":
#     cv = "CalculoIDF/python_scripts/csv/chuvas_C_01844000_CV.csv"
#     pl = "CalculoIDF/python_scripts/csv/chuvas_C_01944009_PL.csv"
#     ma = "CalculoIDF/python_scripts/csv/chuvas_C_02043032_MA.csv"
#     csv_file_path = cv
#     main(csv_file_path)
