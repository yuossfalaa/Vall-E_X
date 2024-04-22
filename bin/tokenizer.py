import logging
import os
from pathlib import Path

import torch
import torch.multiprocessing
from lhotse import CutSet, NumpyHdf5Writer
from lhotse.recipes.utils import read_manifests_if_cached
from tqdm import tqdm

from data.tokenizer import AudioTokenExtractor, AudioTokenConfig
from macros import lang2token
from utils.g2p import PhonemeBpeTokenizer

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Torch's multithreading behavior needs to be disabled, or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.multiprocessing.set_sharing_strategy("file_system")


def Tokenize(src_dir, output_dir, prefix, dataset_parts: list, suffix="jsonl.gz", batch_duration=400.0,
             language='ar', trim_to_supervisions=False):
    assert len(dataset_parts) >= 1

    manifests = read_manifests_if_cached(
        dataset_parts=dataset_parts,
        output_dir=src_dir,
        prefix=prefix,
        suffix=suffix,
        types=["recordings", "supervisions", "cuts"],
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    num_jobs = min(32, os.cpu_count())
    logging.info(f"dataset_parts: {dataset_parts} manifests {len(manifests)}")

    if prefix and not prefix.endswith("_"):
        prefix = f"{prefix}_"

    text_tokenizer = PhonemeBpeTokenizer(tokenizer_path="./utils/g2p/bpe_69.json")
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
        # AudioTokenizer
        storage_path = (
            f"{output_dir}/{prefix}_encodec_{partition}"
        )
        cut_set = cut_set.resample(24000)
        try:
            if trim_to_supervisions:
                cut_set = cut_set.trim_to_supervisions(keep_overlapping=False)
        except Exception:
            logging.info(f"trim_to_supervisions Failed {Exception}")

        print(cut_set.describe())
        with torch.no_grad():
            cut_set = cut_set.compute_and_store_features_batch(
                extractor=AudioTokenExtractor(AudioTokenConfig()),
                storage_path=storage_path,
                num_workers=num_jobs,
                batch_duration=batch_duration,
                collate=False,
                overwrite=True,
                storage_type=NumpyHdf5Writer)
        # TextTokenizer
        print("Start Text Tokenizing")
        cut_set = cut_set
        lang_token = lang2token[language]
        for c in tqdm(cut_set):
            if c.supervisions[0].custom is None:
                c.supervisions[0].custom = {}
            if len(c.supervisions[0].text) == 0:
                print("Empty Text Found")
                c.supervisions[0].text = "0"
            phoneme_tokens, lang = text_tokenizer.tokenize(
                f"_{lang_token + c.supervisions[0].text + lang_token}".strip())
            c.supervisions[0].custom["tokens"] = {"text": phoneme_tokens}
            if lang:
                c.supervisions[0].custom["lang"] = lang[0]
        cuts_filename = f"{prefix}cuts_{partition}.{suffix}"
        cut_set.to_file(f"{output_dir}/{cuts_filename}")
