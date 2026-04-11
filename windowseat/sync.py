import os

import dotenv
from constants import BUCKET_NAME
from minio import Minio

dotenv.load_dotenv()

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
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

    for item in files_to_upload:
        file_name = os.path.basename(item)
        clean_name = file_name.replace("_windowseat_output", "")
        new_path = os.path.join("ai", clean_name)
        print(f"Uploading {item} to {new_path} in bucket {BUCKET_NAME}")

        client.fput_object(
            bucket_name=BUCKET_NAME, object_name=new_path, file_path=item
        )


def download(output_dir: str, items: list[str]):
    for item in items:
        print(f"Downloading {item} from bucket {BUCKET_NAME} to {output_dir}")
        client.fget_object(
            bucket_name=BUCKET_NAME,
            object_name=f"{item}/{item}.jpg",
            file_path=os.path.join(output_dir, f"{item}.jpg"),
        )
