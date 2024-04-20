import concurrent.futures
import os
from concurrent.futures import ProcessPoolExecutor
from logging import info
import tarfile
from pathlib import Path
import time

from lhotse import RecordingSet, SupervisionSegment, SupervisionSet, fix_manifests, validate_recordings_and_supervisions
from lhotse.recipes.utils import read_manifests_if_cached, manifests_exist
from tqdm import tqdm

recordings = RecordingSet()


def generate_mgb2(download_path="DataSet/mgb2_speech", output_dir="manifests/mgb2"):
    _organize_downloaded_data(download_path)
    _create_manifest(output_dir, download_path)


def _create_manifest(output_dir, download_path):
    global recordings
    dataset_path = download_path + "/dataset/"
    job_num = os.cpu_count()
    corpus_dir = Path(dataset_path)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    dataset_parts = ["dev", "train", "test"]
    manifests = {}

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Maybe the manifests already exist: we can read them and save a bit of preparation time.
        manifests = read_manifests_if_cached(
            dataset_parts=dataset_parts,
            output_dir=output_dir,
            prefix="mgb2",
            suffix="jsonl.gz",
            lazy=True,
        )
    for part in dataset_parts:
        print(f"Processing MGB2 subset: {part}")
        if not os.path.exists(Path(f"{corpus_dir}/{part}/wav")) or not os.path.exists(Path(f"{corpus_dir}/{part}/txt")):
            print(f"Processing MGB2 subset: {part} was skipped due to insufficient data")
            continue
        if manifests_exist(
                part=part, output_dir=output_dir, prefix="mgb2", suffix="jsonl.gz"
        ):
            print(f"MGB2 subset: {part} already prepared - skipping.")
            continue
        output_dir = Path(output_dir)

        # Make Recording
        wav_path = Path(f"{corpus_dir}/{part}/wav")
        recordings = RecordingSet.from_dir(wav_path, pattern='*.wav', num_jobs=job_num)

        # Make Supervisions

        txt_path = Path(f"{corpus_dir}/{part}/txt")
        supervisions_list = []
        with ProcessPoolExecutor(job_num) as ex:
            supervisions_list.extend(
                tqdm(
                    ex.map(process, txt_path.iterdir()),
                    desc="Scanning Text files (*.txt): ",
                )
            )

        All_Count = len(supervisions_list)
        res = [i for i in supervisions_list if i is not None]
        supervisions_list = res
        Non_Count = len(supervisions_list)
        print(f"Found {Non_Count} from all {All_Count}")


        supervisions = SupervisionSet.from_segments(supervisions_list)
        recordings, supervisions = fix_manifests(recordings, supervisions)
        validate_recordings_and_supervisions(recordings, supervisions)

        # saving recordings and supervisions
        recordings.to_file((output_dir / f"mgb2_recordings_{part}.jsonl.gz"))
        supervisions.to_file((output_dir / f"mgb2_supervisions_{part}.jsonl.gz"))

        manifests[part] = {
            "recordings": recordings,
            "supervisions": supervisions,
        }
    return manifests


def process(file_path):
    global recordings
    if file_path.is_file() and file_path.suffix == ".txt":
        file_name = file_path.stem
        if  len(file_name) >= 5 and file_name[-5:] == "_utf8":
            record_id = file_name[:-5]
        else :
            record_id = file_name
        try:
            duration = recordings[record_id].duration
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read().strip()

            return SupervisionSegment(id=record_id, recording_id=record_id, start=0,
                                      duration=duration, channel=0,
                                      text=text)
        except:
            #print(f"record id {record_id} not found")
            return None


def _organize_downloaded_data(download_path):
    _DATA_ARCHIVE_ROOT = download_path + "/archives/"
    _DATA_URL = {
        "test": _DATA_ARCHIVE_ROOT + "mgb2_wav.test.tar.gz",
        "dev": _DATA_ARCHIVE_ROOT + "mgb2_wav.dev.tar.gz",
        "train": [_DATA_ARCHIVE_ROOT + f"mgb2_wav_{x}.train.tar.gz" for x in range(48)],  # we have 48 archives
    }
    _TEXT_URL = {
        "test": _DATA_ARCHIVE_ROOT + "mgb2_txt.test.tar.gz",
        "dev": _DATA_ARCHIVE_ROOT + "mgb2_txt.dev.tar.gz",
        "train": _DATA_ARCHIVE_ROOT + "mgb2_txt.train.tar.gz",
    }

    # Create directories for the merged dataset
    merged_dataset_path = download_path + "/dataset/"
    os.makedirs(merged_dataset_path, exist_ok=True)

    for data_type, tar_file_paths in _DATA_URL.items():
        if data_type == "train":
            for tar_file_path in tar_file_paths:
                # Check if the tar file exists
                if os.path.exists(tar_file_path):
                    # Extract data from each tar file
                    with tarfile.open(tar_file_path, "r:gz") as tar:
                        # Extract the contents to the merged dataset directory
                        tar.extractall(path=download_path)
                    # Delete the extracted tar file
                    os.remove(tar_file_path)
                else:
                    print(f"Warning: Tar file '{tar_file_path}' not found. Skipping.")
        else:
            # Check if the tar file exists
            if os.path.exists(tar_file_paths):
                # Extract data from each tar file
                with tarfile.open(tar_file_paths, "r:gz") as tar:
                    # Extract the contents to the merged dataset directory
                    tar.extractall(path=download_path)
                # Delete the extracted tar file
                os.remove(tar_file_paths)
            else:
                print(f"Warning: Tar file '{tar_file_paths}' not found. Skipping.")

    for data_type, tar_file_path in _TEXT_URL.items():
        # Check if the tar file exists
        if os.path.exists(tar_file_path):
            # Extract data from each tar file
            with tarfile.open(tar_file_path, "r:gz") as tar:
                # Extract the contents to the merged dataset directory
                tar.extractall(path=download_path)
            # Delete the extracted tar file
            os.remove(tar_file_path)
        else:
            print(f"Warning: Tar file '{tar_file_path}' not found. Skipping.")

    print("Merged dataset created successfully.")
