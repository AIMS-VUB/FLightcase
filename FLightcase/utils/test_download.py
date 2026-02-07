"""
Test script for a download
"""

import time
import argparse
import requests
from requests.auth import HTTPBasicAuth


def test_download(save_path, download_url, username, password, max_attempts=60):

    print(f"Downloading {download_url}")
    for attempt in range(max_attempts):
        response = requests.get(
            download_url,
            auth=HTTPBasicAuth(username, password),
            stream=True,
            timeout=30,
        )

        if response.status_code == 404:
            print(f"Attempt {attempt+1}/{max_attempts}: File not ready, retrying...")
            time.sleep(1)
            continue

        if response.status_code == 200:
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"✅ Download successful: {save_path}")
            return

        print("❌ Unexpected status:", response.status_code)
        print(response.text)
        return

    print("❌ Download timed out")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--download_url')
    parser.add_argument('--save_path')
    parser.add_argument('--username')
    parser.add_argument('--password')
    args = parser.parse_args()

    test_download(args.save_path, args.download_url, args.username, args.password)
