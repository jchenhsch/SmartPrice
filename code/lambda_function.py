import json
import boto3
import os
from datetime import datetime
from io import StringIO
import pandas as pd

# Initialize the S3 client
s3_client = boto3.client('s3')
featurestore_client = boto3.client('sagemaker-featurestore-runtime')  # Correct client for put_record

# Maintain state (e.g., processed files) using an external storage like DynamoDB (recommended for distributed processing)
processed_files = set()

def lambda_handler(event, context):
    global processed_files

    # Process each record in the S3 event
    for record in event['Records']:
        bucket_name = record['s3']['bucket']['name']
        object_key = record['s3']['object']['key']
        event_type = record['eventName']
        print(f"Processing event type: {event_type} for file: {object_key}")

        # Skip if already processed
        if object_key in processed_files:
            print(f"Skipping already processed file: {object_key}")
            continue

        try:
            # Fetch file content from S3
            response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
            file_content = response['Body'].read().decode('utf-8')

            # Add file to the processed set
            processed_files.add(object_key)

            # Convert CSV content to DataFrame
            df = pd.read_csv(StringIO(file_content))

            # # Add required columns
            # df['number'] = df.index.astype('string')
            # df['event_time'] = pd.Timestamp.now().strftime('%Y-%m-%dT%H:%M:%SZ')

            # # Drop irrelevant columns if they exist
            # columns_to_drop = [
            #     'waterfront', 'view', 'sqft_living15', 'sqft_lot15', 
            #     'lat', 'long', 'zipcode', 'yr_renovated'
            # ]
            # df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

            # # Convert date column format if it exists
            # if 'date' in df.columns:
            #     df['date'] = df['date'].str[:4].astype(int)

            # # Ensure required columns match the feature group schema
            # required_columns = [
            #     'number', 'event_time', 'date', 'price', 'bedrooms', 
            #     'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 
            #     'condition', 'grade', 'sqft_above', 'sqft_basement', 
            #     'yr_built'
            # ]
            # df = df[required_columns]

            # Prepare data for ingestion
            feature_group_name = 'housing-feature-group-simulation'  # Replace with your feature group name
            records = df.to_dict(orient='records')

            # Ingest data into the feature store
            for record in records:
                response = featurestore_client.put_record(
                    FeatureGroupName=feature_group_name,
                    Record=[{'FeatureName': k, 'ValueAsString': str(v)} for k, v in record.items()]
                )
                print(f"Record ingested: {response}")

            print(f"Successfully ingested {len(df)} rows to the feature store.")

            # Add a timestamp to the file name
            timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
            new_file_name = f"{os.path.splitext(object_key)[0]}_{timestamp}.csv"

            # Save the processed file to another S3 bucket
            s3_client.put_object(
                Bucket='mlo4-res',  # Replace with your destination bucket
                Key=new_file_name,
                Body=file_content
            )
            print(f"File saved to mlo4-res bucket with name: {new_file_name}")

        except Exception as e:
            print(f"Error processing file {object_key} from bucket {bucket_name}: {e}")
            return {
                'statusCode': 500,
                'body': json.dumps(f"Error: {str(e)}")
            }

    return {
        'statusCode': 200,
        'body': json.dumps({'result_bucket': 'mlo4-res', 'res_file_key': new_file_name})
    }


