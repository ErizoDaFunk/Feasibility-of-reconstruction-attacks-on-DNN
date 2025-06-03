import urllib.request
import tarfile
import os

# UNIX
# wget -qO - https://github.com/metaspace2020/offsample/releases/download/0.2/GS.tar.gz | tar -xvz

# Define the base and data directories
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# Dataset URL and paths
url = "https://github.com/metaspace2020/offsample/releases/download/0.2/GS.tar.gz"
filename = "GS.tar.gz"
file_path = os.path.join(DATA_DIR, filename)
extract_path = os.path.join(DATA_DIR, 'GS_original')

def download_and_extract():
    if not os.path.exists(file_path):
        print("📥 Downloading:", url)
        urllib.request.urlretrieve(url, file_path)
        print(f"✅ Download complete: {file_path}")
    else:
        print("✅ File already downloaded.")

    if not os.path.exists(extract_path):
        print("📦 Extracting tar.gz file...")
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=extract_path)
        print(f"✅ Extraction complete at: {extract_path}")
    else:
        print("✅ Dataset already extracted.")

if __name__ == "__main__":
    download_and_extract()
