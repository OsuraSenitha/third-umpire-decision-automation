import yaml
from ultralytics import YOLO
import os

if __name__ == "__main__":
    data_dir = os.environ["SM_CHANNEL_TRAIN"]
    model_output_dir = os.environ["SM_MODEL_DIR"]
    intermediate_data_dir = os.environ["SM_OUTPUT_INTERMEDIATE_DIR"]
    output_data_dir = os.environ["SM_OUTPUT_DATA_DIR"]

    # read the data configuration and create the yaml file

    dataset_config = {
        "train": f"{data_dir}/images/train",
        "val": f"{data_dir}/images/val",
        "names": ["Batsman", "Wicket"],
        "nc": 2,
    }
    ds_config_path = f"{data_dir}/dataset.yaml"
    with open(ds_config_path, "w") as handler:
        yaml.dump(dataset_config, handler)

    patience = 50
    epochs = 500

    model = YOLO("yolov8n.pt")
    model.train(
        data=ds_config_path,
        epochs=epochs,
        patience=patience,
        pretrained=True,
        project=output_data_dir,
        name="cricket-object-detect",
        optimizer="Adam",
    )
