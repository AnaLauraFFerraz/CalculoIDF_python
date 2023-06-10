import json
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
        return json.dumps(insufficient_data)
    no_outlier = teste_outlier(processed_data, gb_test)
    distribution_data, params, dist_r2 = distributions(
        no_outlier, table_yn_sigman)
    disaggregation_data, time_interval = disaggregation_coef()
    k_coefficient_data = k_coefficient(params, dist_r2)
    output = ventechow(distribution_data, k_coefficient_data,
                       disaggregation_data, params, time_interval, dist_r2)
    print(output)
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
    bucket_name, blob_name = parse_gcs_url(csv_file_url)
    delete_blob(bucket_name, blob_name)

    return jsonify(result)


def download_csv_file(gcs_url):
    """Download a CSV file from Firebase Cloud Storage to the local machine."""
    if not gcs_url.startswith("gs://"):
        raise ValueError("URL must start with 'gs://'")

    bucket_name, blob_name = parse_gcs_url(gcs_url)

    # Create a Cloud Storage client
    storage_client = storage.Client()

    # Get the bucket
    bucket = storage_client.bucket(bucket_name)

    # Download the blob
    blob = bucket.blob(blob_name)
    csv_file_path = "/tmp/" + blob_name
    blob.download_to_filename(csv_file_path)

    return csv_file_path



def parse_gcs_url(gcs_url):
    """Parse a GCS URL into (bucket_name, blob_name)."""
    # Remove the 'gs://' prefix
    gcs_url = gcs_url[5:]

    # Split the URL into bucket_name and blob_name
    bucket_name, blob_name = gcs_url.split('/', 1)

    return bucket_name, blob_name


def download_blob(bucket_name, blob_name, destination_file_name):
    """Download a blob from a GCS bucket to a local file."""
    # Create a Cloud Storage client
    storage_client = storage.Client()

    # Get the bucket and blob
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    # Download the blob to a local file
    blob.download_to_filename(destination_file_name)


def delete_blob(bucket_name, blob_name):
    """Delete a blob from a GCS bucket."""
    # Create a Cloud Storage client
    storage_client = storage.Client()

    # Get the bucket
    bucket = storage_client.bucket(bucket_name)

    # Delete the blob
    blob = bucket.blob(blob_name)
    blob.delete()


# if __name__ == "__main__":
#     cv = "CalculoIDF_python/src/csv/chuvas_C_01844000_CV.csv"
#     pl = "CalculoIDF_python/src/csv/chuvas_C_01944009_PL.csv"
#     ma = "CalculoIDF_python/src/csv/chuvas_C_02043032_MA.csv"
#     csv_file_path = cv
#     main(csv_file_path)
