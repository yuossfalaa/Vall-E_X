from lhotse.serialization import load_manifest_lazy_or_eager


def copy(input_manifest, output_manifest):
    """
    Load INPUT_MANIFEST and store it to OUTPUT_MANIFEST.
    Useful for conversion between different serialization formats (e.g. JSON, JSONL, YAML).
    Automatically supports gzip compression when '.gz' suffix is detected.
    """
    data = load_manifest_lazy_or_eager(input_manifest)
    data.to_file(output_manifest)
