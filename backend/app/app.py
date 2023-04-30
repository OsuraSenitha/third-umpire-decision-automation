import boto3
import os
from ml.pipeline import Pipeline
from util.file import S3Downloader

OBJ_DECT_WEIGHTS_NAME = os.environ["OBJ_DECT_WEIGHTS_NAME"]
IM_SEG_WEIGHTS_NAME = os.environ["IM_SEG_WEIGHTS_NAME"]

MODELS_PATH = "/tmp/model-weights"
downloader = S3Downloader()
downloader.download(
    "s3://third-umpire-decision-automation-osura/model-weights", MODELS_PATH
)
OBJ_DECT_MODEL_PATH = f"{MODELS_PATH}/{OBJ_DECT_WEIGHTS_NAME}"
IM_SEG_MODEL_PATH = f"{MODELS_PATH}/{IM_SEG_WEIGHTS_NAME}"

s3_client = boto3.client("s3")
BUCKET_NAME = "third-umpire-decision-automation-osura"

pipe = Pipeline(
    object_detect_model_path=OBJ_DECT_MODEL_PATH,
    image_segment_model_path=IM_SEG_MODEL_PATH,
)


def handler(event, context):
    print(event)

    img_key = event["imgKey"]
    filename = os.path.basename(img_key)
    img_path = f"/tmp/{filename}"

    s3_client.download_file(Bucket=BUCKET_NAME, Key=img_key, Filename=img_path)
    results = pipe(
        img_path, batsman_analysis_image_path="/tmp/results/batsman-analysis.jpg"
    )

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": results,
    }
