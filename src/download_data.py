import os
import sys
import requests
import re
from config import *

def get_drive_file_id(url):
    # Extracts the file ID from the Google Drive link
    patterns = [
        r"/d/([\w-]+)",
        r"id=([\w-]+)",
        r"file/d/([\w-]+)"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError("It was not possible to extract the file ID from the provided link.")

def download_file_from_google_drive(file_id, destination):
    # Downloads the file from Google Drive using the ID
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

def main():
    file_id = get_drive_file_id(DATA_SOURCE)
    # Tries to get the file name
    response = requests.get(f"https://drive.google.com/uc?export=download&id={file_id}", stream=True)
    content_disp = response.headers.get('content-disposition')
    if content_disp:
        fname_match = re.findall('filename=\"(.+?)\"', content_disp)
        filename = fname_match[0] if fname_match else f"{file_id}.csv"
    else:
        filename = f"{file_id}.csv"
    destination = os.path.join(DATASET_PATH)
    print(f"Downloading to: {destination}")
    download_file_from_google_drive(file_id, destination)
    print("Download completed!")

if __name__ == "__main__":
    main()
