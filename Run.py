from lhotse.recipes import mgb2 as mgb
import os
import json
import gzip
from utils.g2p import PhonemeBpeTokenizer

if __name__ == '__main__':
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
    # mgb.prepare_mgb2(corpus_dir="./DataSet/mgb2_dev", output_dir="./manifests",buck_walter=True,text_cleaning=True)
    with gzip.open("manifests/mgb2_supervisions_dev.jsonl.gz", 'rt', encoding='utf-8') as fin:  # Open in text mode
        for line in fin:
            json_obj = json.loads(line)
            print(json_obj)

