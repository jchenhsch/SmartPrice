import boto3
import tarfile
import pandas as pd
import awswrangler as wr
from sklearn.model_selection import train_test_split
import h2o
from h2o.automl import H2OAutoML
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset, RegressionPreset
import os
from typing import List

def list_parquet_files(bucket: str, prefix: str) -> List[str]:
    """
    List all parquet files under a given S3 prefix, including date-partitioned folders
    """
    s3_client = boto3.client('s3')
    parquet_files = []
    bucket = bucket.replace('s3://', '')
    paginator = s3_client.get_paginator('list_objects_v2')

    year_pages = paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter='/')
    for year_page in year_pages:
        if 'CommonPrefixes' in year_page:
            for year_prefix in year_page['CommonPrefixes']:
                year_path = year_prefix['Prefix']
                month_pages = paginator.paginate(Bucket=bucket, Prefix=year_path, Delimiter='/')
                for month_page in month_pages:
                    if 'CommonPrefixes' in month_page:
                        for month_prefix in month_page['CommonPrefixes']:
                            month_path = month_prefix['Prefix']
                            day_pages = paginator.paginate(Bucket=bucket, Prefix=month_path, Delimiter='/')
                            for day_page in day_pages:
                                if 'CommonPrefixes' in day_page:
                                    for day_prefix in day_page['CommonPrefixes']:
                                        day_path = day_prefix['Prefix']
                                        file_pages = paginator.paginate(Bucket=bucket, Prefix=day_path)
                                        for file_page in file_pages:
                                            if 'Contents' in file_page:
                                                for obj in file_page['Contents']:
                                                    if obj['Key'].endswith('.parquet'):
                                                        parquet_files.append(f"s3://{bucket}/{obj['Key']}")
    return parquet_files

def read_all_parquets(s3_path: str) -> pd.DataFrame:
    """
    Read and concatenate all parquet files from a given S3 path with date partitions
    """
    s3_path = s3_path.rstrip('/')
    bucket = s3_path.split('/')[2]
    prefix = '/'.join(s3_path.split('/')[3:]) + '/'

    parquet_files = list_parquet_files(bucket, prefix)
    if not parquet_files:
        raise ValueError(f"No parquet files found in {s3_path}")

    df = wr.s3.read_parquet(path=parquet_files)
    return df


def prepare_monitoring_data(train_data, test_data, predictions, aml, target_column='price'):
    """
    Prepare data for Evidently monitoring
    """
    reference_data = train_data.copy()
    current_data = test_data.copy()

    # Convert to H2OFrame for prediction
    reference_h2o = h2o.H2OFrame(reference_data)
    reference_predictions = aml.leader.predict(reference_h2o)
    reference_predictions = reference_predictions.as_data_frame()
    
    reference_data['prediction'] = reference_predictions['predict'].values
    current_data['prediction'] = predictions

    reference_data = reference_data.rename(columns={target_column: 'target'})
    current_data = current_data.rename(columns={target_column: 'target'})

    columns_to_keep = list(reference_data.columns)
    reference_data = reference_data[columns_to_keep]
    current_data = current_data[columns_to_keep]

    return reference_data, current_data

def create_monitoring_report(reference_data, current_data, s3_bucket='mlo4-res', s3_prefix='evidently'):
    """
    Generate comprehensive monitoring report using Evidently and save to S3
    """
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset, RegressionPreset
    import os
    
    s3 = boto3.client('s3')
    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
        TargetDriftPreset()
    ])

    report.run(reference_data=reference_data, current_data=current_data)

    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    report_path = f'monitoring_report_{timestamp}.html'

    report.save_html(report_path)
    s3.upload_file(report_path, s3_bucket, f'{s3_prefix}/monitoring_report_{timestamp}.html')
    os.remove(report_path)
    print(f"Report saved to s3://{s3_bucket}/{s3_prefix}/monitoring_report_{timestamp}.html")


# def save_best_model(leaderboard, bucket_name, s3_key):
#     """
#     Compare and save the best model to S3
#     """
#     s3 = boto3.client('s3')
#     try:
#         s3.download_file(bucket_name, s3_key, 'existing_best_model_info.csv')
#         existing_best_model_df = pd.read_csv('existing_best_model_info.csv')
#     except:
#         existing_best_model_df = pd.DataFrame({'model_id': [''], 'rmse': [float('inf')], 'training_time_ms': [float('inf')]})

#     leaderboard_df = leaderboard.as_data_frame()
#     current_best_model_row = leaderboard_df.loc[leaderboard_df['rmse'].idxmin()]
#     current_best_model_id = current_best_model_row['model_id']
#     current_best_rmse = current_best_model_row['rmse']

#     current_best_model = h2o.get_model(current_best_model_id)
#     current_best_training_time = current_best_model._model_json['output']['run_time']

#     if current_best_rmse < existing_best_model_df['rmse'].iloc[0]:
#         new_best_model_df = pd.DataFrame({
#             'model_id': [current_best_model_id],
#             'rmse': [current_best_rmse],
#             'training_time_ms': [current_best_training_time]
#         })
#         new_best_model_df.to_csv('best_model_info.csv', index=False)
#         s3.upload_file('best_model_info.csv', bucket_name, s3_key)

#         model_path = h2o.save_model(model=current_best_model, path="h2o_models/", force=True)
#         s3.upload_file(Filename=model_path, Bucket=bucket_name, Key=f"housing_automl/h2o_best_model.zip")
#         print(f"Model uploaded to s3://{bucket_name}/housing_automl/h2o_best_model.zip")


def save_best_model(leaderboard, bucket_name, s3_key):
    """
    Compare and save the best model to S3 as a .tar.gz archive
    """
    s3 = boto3.client('s3')
    try:
        s3.download_file(bucket_name, s3_key, 'existing_best_model_info.csv')
        existing_best_model_df = pd.read_csv('existing_best_model_info.csv')
    except:
        existing_best_model_df = pd.DataFrame({'model_id': [''], 'rmse': [float('inf')], 'training_time_ms': [float('inf')]})

    leaderboard_df = leaderboard.as_data_frame()
    current_best_model_row = leaderboard_df.loc[leaderboard_df['rmse'].idxmin()]
    current_best_model_id = current_best_model_row['model_id']
    current_best_rmse = current_best_model_row['rmse']

    current_best_model = h2o.get_model(current_best_model_id)
    current_best_training_time = current_best_model._model_json['output']['run_time']

    if (current_best_rmse < existing_best_model_df['rmse'].iloc[0]) or (existing_best_model_df['rmse'].iloc[0] == 0):
        new_best_model_df = pd.DataFrame({
            'model_id': [current_best_model_id],
            'rmse': [current_best_rmse],
            'training_time_ms': [current_best_training_time]
        })
        new_best_model_df.to_csv('best_model_info.csv', index=False)
        s3.upload_file('best_model_info.csv', bucket_name, s3_key)

        model_path = h2o.save_model(model=current_best_model, path="h2o_models/", force=True)

        # Create a .tar.gz archive of the model directory
        tar_gz_path = 'h2o_best_model.tar.gz'
        with tarfile.open(tar_gz_path, 'w:gz') as tar:
            tar.add(model_path, arcname=os.path.basename(model_path))

        # Upload the .tar.gz file to S3
        s3.upload_file(Filename=tar_gz_path, Bucket=bucket_name, Key=f"housing_automl/h2o_best_model.tar.gz")
        print(f"Model uploaded to s3://{bucket_name}/housing_automl/h2o_best_model.tar.gz")

