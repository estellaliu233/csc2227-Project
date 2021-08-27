import boto3
import math
import dateutil
import json
import os

# grab environment variables
ENDPOINT_NAME = 'xgboost-v1'
client = boto3.client(service_name='sagemaker-runtime')


def transform_data(data):
    try:
        features = data.copy()

        # Return the transformed data. skip datetime field
        return ','.join([str(feature) for feature in features[1:]])

    except Exception as err:
        print('Error when transforming: {0},{1}'.format(data, err))
        raise Exception('Error when transforming: {0},{1}'.format(data, err))


def lambda_handler(event, context):
    try:
        print("Received event: " + json.dumps(event, indent=2))

        request = json.loads(json.dumps(event))

        transformed_data = [transform_data(instance['features']) for instance in request["instances"]]

        # XGBoost accepts data in CSV. It does not support JSON.
        # So, we need to submit the request in CSV format
        # Prediction for multiple observations in the same call
        result = client.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                        Body=('\n'.join(transformed_data).encode('utf-8')),
                                        ContentType='text/csv')

        result = result['Body'].read().decode('utf-8')

        # Apply inverse transformation to get the rental count
        print(result)
        result = result.split(',')
        predictions = [math.expm1(float(r)) for r in result]

        return {
            'statusCode': 200,
            'isBase64Encoded': False,
            'body': json.dumps(predictions)
        }

    except Exception as err:
        return {
            'statusCode': 400,
            'isBase64Encoded': False,
            'body': 'Call Failed {0}'.format(err)
        }