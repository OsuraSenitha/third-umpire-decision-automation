import os
import json

def download_cric_semantic():
  os.system("!pip install -q kaggle")
  # add kaggle credentials
  with open("/content/drive/MyDrive/.kaggle/kaggle.json") as handler:
    kaggle_creds = json.load(handler)
    os.environ['KAGGLE_USERNAME'] = kaggle_creds["username"]
    os.environ['KAGGLE_KEY'] = kaggle_creds["key"]
  os.system("kaggle datasets download -d sadhliroomyprime/cricket-semantic-segmentation -p ./data")
  os.system("unzip ./data/cricket-semantic-segmentation.zip -d ./data")
  os.system("rm -r -f ./data/cricket-semantic-segmentation")
  os.system('mv "./data/www.acmeai.tech ODataset 4 - Cricket Semantic Segmentation" ./data/cricket-semantic-segmentation')
  os.system("rm -r -f /content/data/cricket-semantic-segmentation.zip")
  os.system('rm -f "/content/data/www.acmeai.tech ODataset 4 - Cricket Semantic Segmentation.pdf"')

def download(dataset):
  datasets = ["cricket-semantic"]

  if not os.path.exists("./data"):
    os.makedirs("./data")

  if dataset == datasets[0]:
    download_cric_semantic()  
  else:
    raise ValueError(f"Invalid dataset specification. Allowed dataset names are {datasets}")