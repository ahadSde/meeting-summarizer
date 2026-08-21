from app.services.summarizer import _split_for_backoff, chunk_transcript


def test_short_transcript_does_not_chunk() -> None:
    assert chunk_transcript("[00:00:01] Hello") == ["[00:00:01] Hello"]


def test_long_transcript_chunks_at_line_boundary() -> None:
    transcript = "first line\nsecond line\nthird line"
    chunks = chunk_transcript(transcript, max_chars=16)
    assert chunks == ["first line\n", "second line\n", "third line\n"]


def test_backoff_prefers_a_transcript_line_boundary() -> None:
    text = ("a" * 1000) + "\n" + ("b" * 2000) + "\n"
    left, right = _split_for_backoff(text)
    assert left.endswith("\n")
    assert left + right == text
