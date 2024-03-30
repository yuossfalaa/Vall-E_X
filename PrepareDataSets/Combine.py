from lhotse.serialization import load_manifest_lazy_or_eager
from lhotse.utils import Pathlike


def combine(manifests: list[Pathlike], output_manifest: Pathlike):
    """Load MANIFESTS, combine them into a single one, and write it to OUTPUT_MANIFEST."""
    from lhotse.manipulation import combine as combine_manifests

    data_set = combine_manifests(*[load_manifest_lazy_or_eager(m) for m in manifests])
    data_set.to_file(output_manifest)