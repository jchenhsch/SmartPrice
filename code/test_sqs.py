import boto3
import time
import h2o
from h2o.automl import H2OAutoML
import awswrangler as wr
from sklearn.model_selection import train_test_split  # Fixed this line
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset, RegressionPreset
from auto_ML import read_all_parquets, prepare_monitoring_data, create_monitoring_report, save_best_model

# ingest test script, ingest when lambda successful trigger
# reset the delay to 5-10 minutes for vis period to allow retrain process
# make sure we have a delayed delivery to account for delays updating the offline feature stores

# Initialize the SQS client
sqs = boto3.client('sqs')

# Specify your queue URL
queue_url = 'https://sqs.us-east-2.amazonaws.com/637423203755/success_trigger'

# Long polling duration (up to 20 seconds)
long_polling_wait_time = 20

# Loop to continuously listen for messages
while True:
    # Receive a message from the SQS queue using long polling
    response = sqs.receive_message(
        QueueUrl=queue_url,
        MaxNumberOfMessages=1,
        WaitTimeSeconds=long_polling_wait_time  # Long polling timeout
    )

    # If there are messages in the queue
    if 'Messages' in response:
        # Process each message
        
        for message in response['Messages']:
            print(f"Received message: {message['Body']}")
            h2o.init()
            try:
                s3_train_path = "s3://mlo-team4/features/simulation/637423203755/sagemaker/us-east-2/offline-store/housing-feature-group-simulation-1733973023/data/"
                train_data = read_all_parquets(s3_train_path)

                s3_test_path = "s3://mlo-team4/features/test/637423203755/sagemaker/us-east-2/offline-store/housing-feature-group-test-1733972554/data/"
                test_data = read_all_parquets(s3_test_path)

                train_data, validation_data = train_test_split(train_data, test_size=0.2, random_state=42)

                columns_to_drop = ['Unnamed: 0', 'number', 'event_time', 'write_time', 'api_invocation_time', 'is_deleted']
                train_data = train_data.drop(columns=columns_to_drop, errors='ignore')
                validation_data = validation_data.drop(columns=columns_to_drop, errors='ignore')
                test_data = test_data.drop(columns=columns_to_drop, errors='ignore')

                train_h2o = h2o.H2OFrame(train_data)
                validation_h2o = h2o.H2OFrame(validation_data)
                test_h2o = h2o.H2OFrame(test_data)

                target = 'price'
                features = [col for col in train_data.columns if col != target]

                aml = H2OAutoML(max_models=20, seed=42)
                aml.train(x=features, y=target, training_frame=train_h2o, validation_frame=validation_h2o)

                leaderboard = aml.leaderboard
                print(leaderboard)

                predictions = aml.leader.predict(test_h2o)
                pred_df = predictions.as_data_frame()

                reference_data, current_data = prepare_monitoring_data(
                    train_data=train_h2o.as_data_frame(),
                    test_data=test_h2o.as_data_frame(),
                    predictions=pred_df['predict'].values,
                    aml=aml  # Pass the aml object
                )

                create_monitoring_report(
                    reference_data=reference_data,
                    current_data=current_data
                )

                save_best_model(leaderboard, 'mlo4-res', 'housing_automl/best_model_info.csv')

            except Exception as e:
                print(f"Error: {str(e)}")
                 
            sqs.delete_message(
                QueueUrl=queue_url,
                ReceiptHandle=message['ReceiptHandle']
            )

    else:
        print("No messages in the queue.")
        time.sleep(5)  # Optional sleep to reduce unnecessary requests