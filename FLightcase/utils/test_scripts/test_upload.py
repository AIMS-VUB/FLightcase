"""
Test script for an upload
"""

import os
import argparse
import requests
from requests.auth import HTTPBasicAuth


def test_upload(local_file_path, upload_url, username, password):
    if not os.path.exists(local_file_path):
        print(f"File does not exist: {local_file_path}")
        return

    filename = os.path.basename(local_file_path)

    print(f"Uploading {filename} to {upload_url}")

    with open(local_file_path, "rb") as f:
        response = requests.post(
            upload_url,
            files={"file": f},
            auth=HTTPBasicAuth(username, password),
            timeout=60,
        )

    print("Status:", response.status_code)
    print("Response:", response.text)

    if response.status_code == 201:
        print("✅ Upload successful")
    else:
        print("❌ Upload failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--upload_url')
    parser.add_argument('--local_file_path')
    parser.add_argument('--username')
    parser.add_argument('--password')
    args = parser.parse_args()

    test_upload(args.local_file_path, args.upload_url, args.username, args.password)
