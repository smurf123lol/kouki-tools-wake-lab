from wake_lab.artifacts import ArtifactStore


def test_cas_and_versioned_manifest(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")

    a = store.put_bytes(b"abc", name="a.bin")
    b = store.put_bytes(b"abc", name="b.bin")
    assert a.cas_hash == b.cas_hash
    assert store.get(a) == b"abc"

    ref = store.commit_manifest([{"x": 1}], version=1)
    assert ref.name == "dataset_v1.jsonl"
    assert store.next_version() == 2
