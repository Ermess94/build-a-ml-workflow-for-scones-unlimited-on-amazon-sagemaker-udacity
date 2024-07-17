""" 
The first lambda function is responsible for data generation.
"""

import json
import boto3
import base64

s3 = boto3.client('s3')


def lambda_handler(event, context):
    key = event["s3_key"]
    bucket = event["s3_bucket"]

    # Download the image from S3 to /tmp/image.png
    s3.download_file(bucket, key, "/tmp/image.png")

    # Read and encode the image data
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')

    # Return the serialized image data
    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        }
    }


""" 
The second one is responsible for image classification.
"""


ENDPOINT_NAME = 'image-classification-2024-07-16-14-42-10-266'
runtime_client = boto3.client('runtime.sagemaker')


def lambda_handler(event, context):
    # Decode the image data
    image = base64.b64decode(event["body"]["image_data"])

    # Invoke the SageMaker endpoint
    response = runtime_client.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType='image/png',
        Body=image
    )

    # Parse the response
    inferences = json.loads(response['Body'].read().decode('utf-8'))

    # Return the data back to the Step Function
    event["inferences"] = inferences
    return {
        'statusCode': 200,
        'body': event
    }


""" 
The third function is responsible for filtering out low-confidence inferences.
"""


THRESHOLD = 0.9


def lambda_handler(event, context):
    inferences = event["body"]["inferences"]

    # Check if any inferences meet the threshold
    meets_threshold = any(confidence > THRESHOLD for confidence in inferences)

    if not meets_threshold:
        raise Exception("THRESHOLD_CONFIDENCE_NOT_MET")
    return {
        'statusCode': 200,
        'body': event
    }
