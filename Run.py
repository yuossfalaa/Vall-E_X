from lhotse import CutSet, load_manifest_lazy, load_manifest
from lhotse.bin import lhotse
from lhotse.lazy import LazyManifestIterator
from lhotse.recipes.utils import read_manifests_if_cached

from data import TextTokenCollater, get_text_token_collater
from macros import lang2token

from utils.g2p import PhonemeBpeTokenizer

if __name__ == '__main__':

    print("shuffling train cuts")
    manifests = load_manifest("manifests/output/cuts_train.jsonl.gz")
    manifests = manifests.shuffle(buffer_size=500_000)
    manifests.to_file("manifests/output/cuts_train.jsonl.gz")
    print(manifests[0])
    print(manifests[1])
    print(manifests[2])
    print(manifests[3])



    print("\nshuffling dev cuts")
    manifests = load_manifest("manifests/output/cuts_dev.jsonl.gz")
    manifests = manifests.shuffle(buffer_size=100_000)
    manifests.to_file("manifests/output/cuts_dev.jsonl.gz")
    print(manifests[0])
    print(manifests[1])
    print(manifests[2])
    print(manifests[3])

    print("\nshuffling test cuts")
    manifests = load_manifest("manifests/output/cuts_test.jsonl.gz")
    manifests = manifests.shuffle(buffer_size=100_000)
    manifests.to_file("manifests/output/cuts_test.jsonl.gz")
    print(manifests[0])
    print(manifests[1])
    print(manifests[2])
    print(manifests[3])



