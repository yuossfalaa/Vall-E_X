# from lhotse.recipes import mgb2
import os

from bin.tokenizer import Tokenize
from prepare_mgb2_dev_only_edit import prepare_mgb2

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

if __name__ == '__main__':
    try:
        prepare_mgb2(corpus_dir="DataSet/mgb2_dev", output_dir="manifests", buck_walter=False)
    except:
        pass
    Tokenize("manifests", "manifests/output", "mgb2", ['dev'])


