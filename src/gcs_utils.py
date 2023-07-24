from google.cloud import storage

# Create a Cloud Storage client
storage_client = storage.Client()

def get_bucket_and_blob(gcs_url):
    """Parse a GCS URL into (bucket, blob)."""
    # Remove the 'gs://' prefix
    gcs_url = gcs_url[5:]

    # Split the URL into bucket_name and blob_name
    bucket_name, blob_name = gcs_url.split('/', 1)

    # Get the bucket and blob
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    return bucket, blob

def download_csv_file(gcs_url):
    """Download a CSV file from Firebase Cloud Storage to the local machine."""
    if not gcs_url.startswith("gs://"):
        raise ValueError("URL must start with 'gs://'")

    _, blob = get_bucket_and_blob(gcs_url)
    csv_file_path = "/tmp/" + blob.name
    blob.download_to_filename(csv_file_path)

    return csv_file_path

def delete_blob(gcs_url):
    """Delete a blob from a GCS bucket."""
    _, blob = get_bucket_and_blob(gcs_url)
    blob.delete()
