import os
import logging

from pathlib import Path

from lhotse.recipes.utils import read_manifests_if_cached
from lhotse import CutSet, NumpyHdf5Writer

import torch
import torch.multiprocessing


os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.multiprocessing.set_sharing_strategy("file_system")


def Tokenize(src_dir, output_dir, prefix, dataset_parts: list, suffix="jsonl.gz", batch_duration=400.0):

    assert len(dataset_parts) >= 1

    manifests = read_manifests_if_cached(
        dataset_parts=dataset_parts,
        output_dir=src_dir,
        prefix=prefix,
        suffix=suffix,
        types=["recordings", "supervisions", "cuts"],
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    unique_symbols = set()
    num_jobs = min(32, os.cpu_count())

    logging.info(f"dataset_parts: {dataset_parts} manifests {len(manifests)}")

    if prefix and not prefix.endswith("_"):
        prefix = f"{prefix}_"

    for partition, m in manifests.items():
        logging.info(
            f"Processing partition: {partition} CUDA: {torch.cuda.is_available()}"
        )
        try:
            cut_set = CutSet.from_manifests(
                recordings=m["recordings"],
                supervisions=m["supervisions"],
            )
        except Exception:
            cut_set = m["cuts"]
