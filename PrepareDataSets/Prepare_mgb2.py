import os

from Copy import copy
from lhotse.recipes import mgb2

from bin.tokenizer import Tokenize

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

if __name__ == "__main__":
    dataset_parts = ["dev", "train", "test"]
    try:
        # Results :
        # manifests/mgb2_recordings_dev.jsonl.gz
        # manifests/mgb2_recordings_train.jsonl.gz
        # manifests/mgb2_recordings_test.jsonl.gz
        # manifests/mgb2_supervisions_dev.jsonl.gz
        # manifests/mgb2_supervisions_train.jsonl.gz
        # manifests/mgb2_supervisions_test.jsonl.gz
        mgb2.prepare_mgb2(corpus_dir="Dataset/mgb2", output_dir="manifests", buck_walter=False, text_cleaning=False)

        # Results :
        # manifests/output/mgb2_cuts_dev.jsonl.gz
        # manifests/output/mgb2_cuts_train.jsonl.gz
        # manifests/output/mgb2_cuts_test.jsonl.gz
        # manifests/output/mgb2__encodec_dev.h5
        # manifests/output/mgb2__encodec_train.h5
        # manifests/output/mgb2__encodec_test.h5
        Tokenize(src_dir="manifests", output_dir="manifests/output", prefix="mgb2", dataset_parts=dataset_parts,
                 language='ar')

        # Results :
        # manifests/output/cuts_dev.jsonl.gz
        # manifests/output/cuts_train.jsonl.gz
        # manifests/output/cuts_test.jsonl.gz
        # This Result Will be used by the model , the names are Fixed for all Datasets , and shouldn't be changed
        copy("manifests/output/mgb2_cuts_dev.jsonl.gz", "manifests/output/cuts_dev.jsonl.gz")
        copy("manifests/output/mgb2_cuts_train.jsonl.gz", "manifests/output/cuts_train.jsonl.gz")
        copy("manifests/output/mgb2_cuts_test.jsonl.gz", "manifests/output/cuts_test.jsonl.gz")

    except Exception:
        print(Exception)
