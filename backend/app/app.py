import boto3
import os, traceback
from ml.pipeline import Pipeline
from util.file import S3Downloader, OutputProcessor

OBJ_DECT_WEIGHTS_NAME = os.environ["OBJ_DECT_WEIGHTS_NAME"]
IM_SEG_WEIGHTS_NAME = os.environ["IM_SEG_WEIGHTS_NAME"]
CLASSIFY_WEIGHTS_NAME = os.environ["CLASSIFY_WEIGHTS_NAME"]
BUCKET_NAME = os.environ["BUCKET_NAME"]
OUTPUT_KEY = os.environ["OUTPUT_KEY"]

MODELS_PATH = "/tmp/model-weights"
downloader = S3Downloader()
downloader.download(
    "s3://third-umpire-decision-automation-osura/model-weights", MODELS_PATH
)
OBJ_DECT_MODEL_PATH = f"{MODELS_PATH}/{OBJ_DECT_WEIGHTS_NAME}"
IM_SEG_MODEL_PATH = f"{MODELS_PATH}/{IM_SEG_WEIGHTS_NAME}"
CLASSIFY_MODEL_PATH = f"{MODELS_PATH}/{CLASSIFY_WEIGHTS_NAME}"

s3_client = boto3.client("s3")

pipe = Pipeline(
    object_detect_model_path=OBJ_DECT_MODEL_PATH,
    image_segment_model_path=IM_SEG_MODEL_PATH,
    classifier_model_path=CLASSIFY_MODEL_PATH,
)
output_processor = OutputProcessor(s3_client, BUCKET_NAME, OUTPUT_KEY)


def handler(event, context):
    if "imgKey" in event.keys():
        try:
            img_key = event["imgKey"]
            filename = os.path.basename(img_key)
            img_path = f"/tmp/{filename}"

            s3_client.download_file(Bucket=BUCKET_NAME, Key=img_key, Filename=img_path)
            results = pipe(
                img_path,
                batsman_analysis_image_path="/tmp/results/batsman-analysis.jpg",
                wicket_img_path="/tmp/results/wicket.jpg"
            )
            results = output_processor(results)

            return {
                "statusCode": 200,
                "headers": {"Content-Type": "application/json"},
                "body": results,
            }
        except Exception as e:
            traceback.print_exc()
            return {
                "statusCode": 500,
                "headers": {"Content-Type": "application/json"},
                "body": {"message": "Unknown server error"},
            }
    else:
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json"},
            "body": {"message": "'imgKey' was not found in the request body"},
        }
