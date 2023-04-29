import boto3
import shutil, os
from pathlib import Path
from tqdm.auto import tqdm


class S3Downloader:
    def __init__(self, creds_dir="/content/drive/MyDrive/.aws"):
        home = str(Path.home())
        if ".aws" not in os.listdir(home):
            if os.path.exists(creds_dir):
                shutil.copytree(creds_dir, f"{home}/.aws")
            else:
                raise FileNotFoundError("AWS Credentials were not found")
        self.client = boto3.client("s3")

    def _is_dir(self, path):
        split = os.path.split(path)
        if "." in split[1] and split[1][0] != ".":
            return False
        return True

    def download(self, s3_uri, download_loc):
        trm_str = s3_uri[5:]
        item_lst = trm_str.split("/")
        bucket_name = item_lst.pop(0)
        obj_key = "/".join(item_lst)
        download_loc = os.path.abspath(download_loc)

        if not self._is_dir(obj_key):
            src_obj_name = os.path.split(obj_key)[1]
            download_path = download_loc
            if self._is_dir(download_path):
                download_path = download_loc + "/" + src_obj_name
            self.client.download_file(bucket_name, obj_key, download_path)

        else:
            if not self._is_dir(download_loc):
                raise ValueError(
                    "If the s3_uri is a directory, download_loc cannot be a file"
                )
            else:
                all_objs = self.client.list_objects(Bucket=bucket_name)["Contents"]
                dir_objs = [
                    obj["Key"] for obj in all_objs if obj["Key"].startswith(obj_key)
                ]
                dir_objs = [
                    dir_obj for dir_obj in dir_objs if not self._is_dir(dir_obj)
                ]
                with tqdm(total=len(dir_objs)) as pbar:
                    for src_path in dir_objs:
                        dst_path = os.path.join(
                            download_loc, src_path.lstrip(obj_key).strip("/")
                        )
                        dst_dir = os.path.split(dst_path)[0]
                        os.makedirs(dst_dir, exist_ok=True)
                        self.client.download_file(bucket_name, src_path, dst_path)
                        pbar.update(1)
