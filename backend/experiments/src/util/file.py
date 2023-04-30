from typing import Dict
import boto3
import shutil, os
from uuid import uuid4
from ..infer.pipeline import PipelineOutput


class S3Downloader:
    def __init__(self, creds_dir=None, home_dir=None):
        if creds_dir is not None and home_dir is not None:
            if os.path.exists(creds_dir):
                shutil.copytree(creds_dir, f"{home_dir}/.aws")
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
                for src_path in dir_objs:
                    dst_path = os.path.join(
                        download_loc, src_path.lstrip(obj_key).strip("/")
                    )
                    dst_dir = os.path.split(dst_path)[0]
                    os.makedirs(dst_dir, exist_ok=True)
                    self.client.download_file(bucket_name, src_path, dst_path)


class OutputProcessor:
    def __init__(
        self,
        s3_client,
        output_bucket: str,
        results_key: str = "results",
    ) -> None:
        self.id = str(uuid4())
        self.s3_client = s3_client
        self.output_bucket = output_bucket
        self.results_key = results_key

    def __call__(self, pipe_results: PipelineOutput) -> Dict:
        src_batsman_img_analysis_path = pipe_results.batsman_analysis_img_path
        dst_batsman_s3_key = (
            self.results_key + "/" + self.id + "/batsman-analysis-img.jpg"
        )
        self.s3_client.upload_file(
            src_batsman_img_analysis_path, self.output_bucket, dst_batsman_s3_key
        )
        dst_batsman_s3_uri = f"s3://{self.output_bucket}/{dst_batsman_s3_key}"
        body = {
            "annotations": pipe_results.annotations,
            "batsman_comment": str(pipe_results.batsman_result),
            "batsman_analysis_img_s3_uri": dst_batsman_s3_uri,
            "wicket_comment": str(pipe_results.wicket_result),
            "job_id": self.id,
        }

        return body
