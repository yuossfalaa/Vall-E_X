import os

from Copy import copy
from lhotse.recipes import libritts

if __name__ == "__main__":
    try:
        libritts.download_libritts(target_dir="Dataset/libritts")
        print("Dataset Downloaded")
    except Exception:
        print(Exception)