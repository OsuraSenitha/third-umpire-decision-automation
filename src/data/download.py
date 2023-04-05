import os
import json

def download_cric_semantic(download_location, drive_mount_point):
  os.system("!pip install -q kaggle")
  # add kaggle credentials
  with open(f"{drive_mount_point}/MyDrive/.kaggle/kaggle.json") as handler:
    kaggle_creds = json.load(handler)
    os.environ['KAGGLE_USERNAME'] = kaggle_creds["username"]
    os.environ['KAGGLE_KEY'] = kaggle_creds["key"]

  final_dir = f"{download_location}/cricket-semantic-segmentation"
  os.system(f"kaggle datasets download -d sadhliroomyprime/cricket-semantic-segmentation -p {download_location}")
  os.system(f"unzip {download_location}/cricket-semantic-segmentation.zip -d {download_location}")
  os.system(f"rm -r -f {final_dir}")
  os.system(f'mv "{download_location}/www.acmeai.tech ODataset 4 - Cricket Semantic Segmentation" {final_dir}')
  os.system(f"rm -r -f {download_location}/cricket-semantic-segmentation.zip")
  os.system(f'rm -f "{download_location}/www.acmeai.tech ODataset 4 - Cricket Semantic Segmentation.pdf"')

  return final_dir

def download(dataset, download_location="./data", drive_mount_point="/content/drive"):
  datasets = ["cricket-semantic"]

  if not os.path.exists(download_location):
    os.makedirs(download_location)

  if dataset == datasets[0]:
    final_dir = download_cric_semantic(download_location, drive_mount_point)  
    return final_dir
  else:
    raise ValueError(f"Invalid dataset specification. Allowed dataset names are {datasets}")