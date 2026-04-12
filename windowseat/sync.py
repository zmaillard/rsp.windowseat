import logging
import os

import dotenv
from constants import BUCKET_NAME
from minio import Minio

logger = logging.getLogger(__name__)

dotenv.load_dotenv()

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
logger.info("Initializing Minio client: endpoint=%s", os.getenv("R2_ENDPOINT", ""))
client = Minio(
    endpoint=os.getenv("R2_ENDPOINT", ""),
    access_key=os.getenv("R2_ACCESS_KEY", ""),
    secret_key=os.getenv("R2_SECRET_KEY", ""),
)


def upload(output: str):
    files_to_upload = [
        os.path.join(output, f)
        for f in os.listdir(output)
        if f.endswith("_windowseat_output.jpg")
    ]
    logger.info(
        "upload: found %d file(s) to upload in %s", len(files_to_upload), output
    )

    for item in files_to_upload:
        file_name = os.path.basename(item)
        clean_name = file_name.replace("_windowseat_output", "")
        new_path = os.path.join("ai", clean_name)
        logger.info("Uploading %s to %s in bucket %s", item, new_path, BUCKET_NAME)

        client.fput_object(
            bucket_name=BUCKET_NAME, object_name=new_path, file_path=item
        )
        logger.info("Upload complete: %s", new_path)


def download(output_dir: str, items: list[str]):
    logger.info("download: %d item(s) to %s", len(items), output_dir)
    for item in items:
        logger.info(
            "Downloading %s from bucket %s to %s", item, BUCKET_NAME, output_dir
        )
        client.fget_object(
            bucket_name=BUCKET_NAME,
            object_name=f"{item}/{item}.jpg",
            file_path=os.path.join(output_dir, f"{item}.jpg"),
        )
        logger.info("Download complete: %s", item)
