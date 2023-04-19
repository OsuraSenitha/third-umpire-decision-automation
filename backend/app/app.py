import boto3
import os
from ml.object_detect import ObjectDetectModel

s3_client = boto3.client("s3")
BUCKET_NAME = "third-umpire-decision-automation-osura"

onnx_path = "ml/best.onnx"
model = ObjectDetectModel(onnx_path)

def handler(event, context):
    print(event)

    audio_key = event["audioKey"]
    filename = os.path.basename(audio_key)
    download_path = f"/tmp/{filename}"

    s3_client.download_file(Bucket=BUCKET_NAME, Key=audio_key, Filename=download_path)
    
    output = model(download_path)
    annotations = output.tolist()

    print(f"Sending annotations: {annotations}")
    
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json"
        },
        "body": {
            "annotations": annotations
        }
    }
