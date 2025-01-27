import os
import boto3
import urllib.request as request
from pathlib import Path
import zipfile
from cnnClassifier import logger
from cnnClassifier.utils.common import get_size
from cnnClassifier.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        self.s3 = boto3.client('s3')
    
    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            s3_url = s3_url = self.config.source_url
            logger.info(f'Source URL: {type(s3_url)}, value: {s3_url}')

            # Ensure the source_url is a string
            if not isinstance(s3_url, str):
                raise ValueError(f"source_url must be a string. Got {type(s3_url)} instead.")


            bucket_name = s3_url.split('/')[2]
            object_key = '/'.join(s3_url.split('/')[3:])

            logger.info(f"Downloading file from s3 bucket: {bucket_name}, object_key: {object_key}")

            with open(self.config.local_data_file, 'wb') as f:
                self.s3.download_fileobj(bucket_name, object_key, f)
            logger.info(f"File downloaded at: {self.config.local_data_file}")
            # filename, headers = request.urlretrieve(
            #     url=self.config.source_url,
            #     filename=self.config.local_data_file
            # )
            # logger.info(f"{filename} download with following info: \n{headers}")
        else:
            logger.info(f'File already exists of size: {get_size(Path(self.config.local_data_file))}')

    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file to the data dir (unzip_dir)
        Function returns None
        """ 
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)