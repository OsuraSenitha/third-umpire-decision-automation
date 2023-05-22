import yaml
from ultralytics import YOLO
import os

if __name__ == "__main__":
    data_dir = os.environ["SM_CHANNEL_TRAIN"]
    model_output_dir = os.environ["SM_MODEL_DIR"]
    intermediate_data_dir = os.environ["SM_OUTPUT_INTERMEDIATE_DIR"]
    output_data_dir = os.environ["SM_OUTPUT_DATA_DIR"]

    patience = 50
    epochs = 5000

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
