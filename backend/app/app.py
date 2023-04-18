from ultralytics import YOLO
import torch
import boto3
import os

model = YOLO('yolov8n.pt')
s3_client = boto3.client("s3")
BUCKET_NAME = "third-umpire-decision-automation-osura"

def handler(event, context):
    print(event)

    audio_key = event["audioKey"]
    filename = os.path.basename(audio_key)
    download_path = f"/tmp/{filename}"

    s3_client.download_file(Bucket=BUCKET_NAME, Key=audio_key, Filename=download_path)
    pred = model.predict(download_path)[0]
    cls = pred.cls
    xywh = pred.xywh
    cls_col = cls.reshape((-1, 1))
    annotations = torch.cat((xywh, cls_col), dim=1).tolist()

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
