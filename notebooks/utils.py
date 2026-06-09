import os
import tarfile
import zipfile
import requests
from pathlib import Path


def _download_url(url, file_path):
  response = requests.get(url, stream=True)
  response.raise_for_status()  # Raise an error for bad status codes
  with open(file_path, "wb") as file:
      for chunk in response.iter_content(chunk_size=8192):
          file.write(chunk)

def download_file(url, file_name, cache_dir="data", extract=True, force_download=False, archive_folder=None):
    # Ensure the cache directory exists
    os.makedirs(cache_dir, exist_ok=True)
    file_path = Path(cache_dir) / file_name

    # Download the file
    if not os.path.exists(file_path) or force_download:
      _download_url(url, file_path)
      print(f"File downloaded to: {file_path}")
    else:
      print(f"File already exists at: {file_path}")

    if extract:
      if file_path.suffixes[-2:] == ['.tar', '.gz']:
        print("Extracting tar.gz file...")
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=cache_dir)
      elif file_path.suffix == ".zip":
        print("Extracting zip file...")
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(cache_dir)
      else:
        raise ValueError("Unsupported file format. Only .tar.gz and .zip files are supported.")
      
      print(f"File extracted to: {cache_dir}")
      return Path(cache_dir) / archive_folder if archive_folder is not None else Path(cache_dir)


    return Path(file_path)