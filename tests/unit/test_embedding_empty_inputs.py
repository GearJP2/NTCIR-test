from services.embedding.clap_encoder import ClapEncoder
from services.embedding.text_encoder import TextEncoder


def test_clap_encoder_empty_batch_returns_empty_list():
    assert ClapEncoder().encode_batch([]) == []


def test_text_encoder_empty_batch_returns_empty_list():
    assert TextEncoder().encode_batch([]) == []
