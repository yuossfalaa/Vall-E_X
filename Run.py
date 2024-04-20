from lhotse import SupervisionSet, CutSet
from lhotse.recipes.utils import read_manifests_if_cached

from PrepareDataSets.Generate_mgb2_from_custom_data import generate_mgb2

if __name__ == '__main__':
    dataset_parts = ["dev", "train", "test"]
    src_dir = "manifests/mgb2"
    output_dir = "manifests/output"
    prefix = "mgb2"
    suffix = "jsonl.gz"
    manifests = read_manifests_if_cached(
        dataset_parts=dataset_parts,
        output_dir=src_dir,
        prefix=prefix,
        suffix=suffix,
        types=["recordings", "supervisions", "cuts"],
    )
    for partition, m in manifests.items():
        cut_set = CutSet.from_manifests(
            recordings=m["recordings"],
            supervisions=m["supervisions"],
        )
        print(cut_set.describe())

