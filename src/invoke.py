import boto3
import json

lambda_client = boto3.client('lambda')

customer = {"customer":
            {"age": 65,
             "gender": "Female",
             "sleep_duration": 6.0,
             "deep_sleep_percentage": 70,
             "light_sleep_percentage": 12,
             "awakenings": 0,
             "caffeine_consumption": 0,
             "alcohol_consumption": 0,
             "smoking_status": "Yes",
             "exercise_frequency": 3.0,
             "sleep_time": 1.0,
             "getup_time": 7
             }
            }

response = lambda_client.invoke(
    FunctionName='sleep-efficiency-docker',
    InvocationType='RequestResponse',
    Payload=json.dumps(customer)
)

result = json.loads(response['Payload'].read())
print(json.dumps(result, indent=2))
