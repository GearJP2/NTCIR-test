from storage.milvus.collections import _ensure


class FakeClient:
    def __init__(self, exists=True, load_error=None):
        self.exists = exists
        self.load_error = load_error
        self.calls = []

    def has_collection(self, name):
        self.calls.append(("has_collection", name))
        return self.exists

    def create_collection(self, **kwargs):
        self.calls.append(("create_collection", kwargs["collection_name"]))

    def create_index(self, **kwargs):
        self.calls.append(("create_index", kwargs["collection_name"]))

    def load_collection(self, name):
        self.calls.append(("load_collection", name))
        if self.load_error is not None:
            error = self.load_error
            self.load_error = None
            raise error


def test_ensure_creates_index_and_loads_new_collection():
    client = FakeClient(exists=False)

    _ensure(client, "visual_keyframes", schema=object(), vec_field="visual_vector", dim=768)

    assert [call[0] for call in client.calls] == [
        "has_collection",
        "create_collection",
        "create_index",
        "load_collection",
    ]


def test_ensure_loads_existing_collection():
    client = FakeClient(exists=True)

    _ensure(client, "visual_keyframes", schema=object(), vec_field="visual_vector", dim=768)

    assert [call[0] for call in client.calls] == ["has_collection", "load_collection"]


def test_ensure_repairs_existing_collection_missing_index():
    client = FakeClient(exists=True, load_error=RuntimeError("index not found"))

    _ensure(client, "visual_keyframes", schema=object(), vec_field="visual_vector", dim=768)

    assert [call[0] for call in client.calls] == [
        "has_collection",
        "load_collection",
        "create_index",
        "load_collection",
    ]


def test_ensure_reraises_non_index_load_errors():
    client = FakeClient(exists=True, load_error=RuntimeError("connection failed"))

    try:
        _ensure(client, "visual_keyframes", schema=object(), vec_field="visual_vector", dim=768)
    except RuntimeError as exc:
        assert str(exc) == "connection failed"
    else:
        raise AssertionError("expected RuntimeError")
