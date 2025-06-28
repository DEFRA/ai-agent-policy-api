from logging import getLogger
from typing import Optional

import boto3

from app.config import config

logger = getLogger(__name__)

class S3Client:
    _instance = None

    def __new__(cls, bucket_name: Optional[str] = None):
        """Create a Singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.bucket_name = bucket_name or config.S3_BUCKET
            cls._instance.s3 = boto3.client(
                "s3",
                endpoint_url=config.S3_ENDPOINT
            )
        return cls._instance

    def check_connection(self):
        """Checks if the connection to the S3 bucket is working."""
        try:
            self.s3.head_bucket(Bucket=self.bucket_name)
            logger.info("Connected to S3 bucket: %s", self.bucket_name)
            return True
        except Exception as e:
            logger.error("Failed to connect to S3 bucket: %s", e)
        return False


    def check_object_existence(self, object_name: str) -> bool:
        """Checks if the named object exists in the attached S3 bucket."""
        try:
            self.s3.head_object(Bucket=self.bucket_name, Key=object_name)
            logger.info("%s exists in %s", object_name, self.bucket_name)
            return True
        except Exception as e:
            logger.error("Failed to find %s in %s: %s", object_name, e, self.bucket_name, e)
        return False

    def upload_file(self, file_name: str, object_name: str = None) -> None:
        """Uploads a file to the S3 bucket."""
        if object_name is None:
            object_name = file_name
        try:
            self.s3.upload_file(file_name, self.bucket_name, object_name)
            logger.info("File %s uploaded as %s", file_name, object_name)
        except Exception as e:
            logger.error("Upload of %s failed: %s", file_name, e)


    def download_file(self, object_name, file_name):
        """Downloads a file from the S3 bucket."""
        try:
            self.s3.download_file(self.bucket_name, object_name, file_name)
            logger.info("File %s downloaded as %s", object_name, file_name)
        except Exception as e:
            logger.error("Download failed for object %s with file name %s : %s", object_name, e, file_name, e)

    def close_connection(self):
        """Closes the S3 client connection."""
        self.s3 = None
        S3Client._instance = None
        logger.info("S3 client connection closed.")
