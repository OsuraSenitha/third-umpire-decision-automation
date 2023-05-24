import yaml
from ultralytics import YOLO
import os
import boto3

if __name__ == "__main__":
    data_dir = os.environ["SM_CHANNEL_TRAIN"]
    model_output_dir = os.environ["SM_MODEL_DIR"]
    intermediate_data_dir = os.environ["SM_OUTPUT_INTERMEDIATE_DIR"]
    output_data_dir = os.environ["SM_OUTPUT_DATA_DIR"]

    patience = 50
    epochs = 50000

    model = YOLO("yolov8n-cls.pt")
    model.train(
        data=data_dir,
        epochs=epochs,
        patience=patience,
        pretrained=True,
        project=output_data_dir,
        name="wicket-classification",
        optimizer="Adam",
        batch=4,
    )

    ses_client = boto3.client("ses", region_name="ap-south-1")
    response = ses_client.send_email(
        Source="pereramat2000@gmail.com",
        Destination={
            "ToAddresses": ["pereramat2000@gmail.com"],
        },
        ReplyToAddresses=["pereramat2000@gmail.com"],
        Message={
            "Subject": {
                "Data": "Wicket Classification Training Complete",
                "Charset": "utf-8",
            },
            "Body": {"Text": {"Data": "Done", "Charset": "utf-8"}},
        },
    )
