from lhotse import CutSet
from lhotse.recipes.utils import read_manifests_if_cached

from PrepareDataSets import Generate_mgb2_from_custom_data as mg
if __name__ == '__main__':
    manifests = read_manifests_if_cached(
        dataset_parts= ["dev"],
        output_dir="manifests/mgb2",
        prefix="mgb2",
        suffix="jsonl.gz",
        types=["recordings", "supervisions", "cuts"],
    )
    for partition, m in manifests.items():
        cut_set = CutSet.from_manifests(
        recordings=m["recordings"],
        supervisions=m["supervisions"],
        )

        print(cut_set.describe())
        print("Hello")


