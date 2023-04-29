import boto3
import os
from ml.object_detect import ObjectDetectModel
from util.file import S3Downloader

SAM_WEIGHTS_NAME = os.environ["SAM_WEIGHTS_NAME"]
YOLO_WEIGHTS_NAME = os.environ["YOLO_WEIGHTS_NAME"]

MODELS_PATH = "/tmp/model-weights"

downloader = S3Downloader(None)
downloader.download(
    "s3://third-umpire-decision-automation-osura/model-weights", MODELS_PATH
)
SAM_MODEL_PATH = f"{MODELS_PATH}/{SAM_WEIGHTS_NAME}"
YOLO_MODEL_PATH = f"{MODELS_PATH}/{YOLO_WEIGHTS_NAME}"

s3_client = boto3.client("s3")
BUCKET_NAME = "third-umpire-decision-automation-osura"

# onnx_path = "ml/best.onnx"
detection_model = ObjectDetectModel(YOLO_MODEL_PATH)


def handler(event, context):
    print(event)

    img_key = event["imgKey"]
    filename = os.path.basename(img_key)
    download_path = f"/tmp/{filename}"

    s3_client.download_file(Bucket=BUCKET_NAME, Key=img_key, Filename=download_path)

    output = detection_model(download_path)
    annotations = output.tolist()

    print(f"Sending annotations: {annotations}")

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": {"annotations": annotations},
    }
