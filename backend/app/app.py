import boto3
import os
from ml.object_detect import ObjectDetectModel
from ml.batsman_segmentor import BatsmanSegmentor
from ml.pipeline import Pipeline
from util.file import S3Downloader

SAM_WEIGHTS_NAME = os.environ["SAM_WEIGHTS_NAME"]
YOLO_WEIGHTS_NAME = os.environ["YOLO_WEIGHTS_NAME"]

MODELS_PATH = "/tmp/model-weights"

downloader = S3Downloader()
downloader.download(
    "s3://third-umpire-decision-automation-osura/model-weights", MODELS_PATH
)
SAM_MODEL_PATH = f"{MODELS_PATH}/{SAM_WEIGHTS_NAME}"
YOLO_MODEL_PATH = f"{MODELS_PATH}/{YOLO_WEIGHTS_NAME}"

s3_client = boto3.client("s3")
BUCKET_NAME = "third-umpire-decision-automation-osura"

detection_model = ObjectDetectModel(YOLO_MODEL_PATH)
segmentation_model = BatsmanSegmentor(SAM_MODEL_PATH)
pipe = Pipeline(detection_model, segmentation_model)


def handler(event, context):
    print(event)

    img_key = event["imgKey"]
    filename = os.path.basename(img_key)
    img_path = f"/tmp/{filename}"

    s3_client.download_file(Bucket=BUCKET_NAME, Key=img_key, Filename=img_path)
    results = pipe(img_path)

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": results,
    }
