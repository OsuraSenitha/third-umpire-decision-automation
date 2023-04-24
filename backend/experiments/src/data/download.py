import os, shutil
import json
from sagemaker.s3 import S3Downloader


def download_cric_semantic(download_location: str, drive_mount_point: str) -> str:
    os.system("!pip install -q kaggle")
    # add kaggle credentials
    with open(f"{drive_mount_point}/MyDrive/.kaggle/kaggle.json") as handler:
        kaggle_creds = json.load(handler)
        os.environ["KAGGLE_USERNAME"] = kaggle_creds["username"]
        os.environ["KAGGLE_KEY"] = kaggle_creds["key"]

    final_dir = f"{download_location}/cricket-semantic-segmentation"
    os.system(
        f"kaggle datasets download -d sadhliroomyprime/cricket-semantic-segmentation -p {download_location}"
    )
    os.system(
        f"unzip {download_location}/cricket-semantic-segmentation.zip -d {download_location}"
    )
    os.system(f"rm -r -f {final_dir}")
    os.system(
        f'mv "{download_location}/www.acmeai.tech ODataset 4 - Cricket Semantic Segmentation" {final_dir}'
    )
    os.system(f"rm -r -f {download_location}/cricket-semantic-segmentation.zip")
    os.system(
        f'rm -f "{download_location}/www.acmeai.tech ODataset 4 - Cricket Semantic Segmentation.pdf"'
    )

    return final_dir


def download_object_detect(download_location: str = "./datasets") -> str:
    temp_path = "./tmp"
    dataset_path = f"{download_location}/cricket-object-detect"
    config_path = "./cricket-object-detect.yaml"

    S3Downloader.download(
        "s3://third-umpire-decision-automation-osura/datasets/cricket-object-detect.zip",
        f"{temp_path}/cricket-object-detect",
    )
    shutil.unpack_archive(
        "./tmp/cricket-object-detect/cricket-object-detect.zip", dataset_path
    )
    shutil.rmtree(temp_path)
    os.system(f"cp {dataset_path}/cricket-object-detect.yaml {config_path}")

    return config_path


def download_batsmen_segmentation(download_location: str = "./datasets") -> str:
    tmp_path = "./tmp"
    dataset_path = f"{download_location}/batsmen-segmentation"
    config_path = "./batsmen-segmentation.yaml"

    S3Downloader.download(
        "s3://third-umpire-decision-automation-osura/datasets/batsmen-segmentation.zip",
        tmp_path,
    )
    shutil.unpack_archive(f"{tmp_path}/batsmen-segmentation.zip", dataset_path)
    shutil.rmtree(tmp_path)
    shutil.copy(f"{dataset_path}/batsmen-segmentation.yaml", config_path)

    return config_path


def download(
    dataset,
    download_location: str = "./datasets",
    drive_mount_point: str = "/content/drive",
) -> str:
    datasets = ["cricket-semantic", "cricket-object-detect", "batsmen-segmentation"]

    if not os.path.exists(download_location):
        os.makedirs(download_location)

    if dataset == datasets[0]:
        final_dir = download_cric_semantic(download_location, drive_mount_point)
        return final_dir
    elif dataset == datasets[1]:
        config_path = download_object_detect(download_location)
        return config_path
    elif dataset == datasets[2]:
        config_path = download_batsmen_segmentation(download_location)
        return config_path
    else:
        raise ValueError(
            f"Invalid dataset specification. Allowed dataset names are {datasets}"
        )
