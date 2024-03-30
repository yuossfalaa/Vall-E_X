import os


from Copy import copy
from lhotse.recipes import libritts

from PrepareDataSets.Combine import combine
from bin.tokenizer import Tokenize

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
if __name__ == "__main__":
    dataset_parts = [
        "dev-clean",
        "dev-other",
        "test-clean",
        "test-other",
        "train-clean-100",
        "train-clean-360",
        "train-other-500",
    ]
    try:
        libritts.prepare_libritts(corpus_dir="Dataset/libritts", output_dir="manifests/libritts")

        Tokenize(src_dir="manifests/libritts", output_dir="manifests/output", prefix="libritts",
                 dataset_parts=dataset_parts, language='en', trim_to_supervisions=False)

        # Prepare LibriTTS "train/dev/test"

        # Train
        train_manifests = [
            "manifests/output/libritts_cuts_train-clean-100.jsonl.gz",
            "manifests/output/libritts_cuts_train-clean-360.jsonl.gz",
            "manifests/output/libritts_cuts_train-other-500.jsonl.gz",
        ]
        combine(manifests=train_manifests,output_manifest="manifests/output/cuts_train.jsonl.gz")

        # dev
        copy(input_manifest="manifests/output/libritts_cuts_dev-clean.jsonl.gz",
             output_manifest="manifests/output/cuts_dev.jsonl.gz")

        # Test
        copy(input_manifest="manifests/output/libritts_cuts_test-clean.jsonl.gz",
             output_manifest="manifests/output/cuts_test.jsonl.gz")
    except Exception:
        print(Exception)
