import json
from unittest.mock import MagicMock, patch

import pytest

from app.services.summarizer import (
    EmptyCompletionError,
    _completion,
    _json_completion,
    _split_for_backoff,
    _summarize_section_with_backoff,
    chunk_transcript,
    summarize_transcript,
)


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


def test_summarize_transcript_missing_api_key_raises() -> None:
    with pytest.raises(RuntimeError, match="GROQ_API_KEY is not configured"):
        summarize_transcript("some transcript", api_key="")


def test_json_completion_normalizes_missing_fields() -> None:
    mock_client = MagicMock()
    # Response JSON missing key_points, decisions, action_items, open_questions
    partial_json = json.dumps({"overview": "Short sync."})

    with patch("app.services.summarizer._completion", return_value=partial_json):
        result = _json_completion(mock_client, "sys_prompt", "user_prompt", "model", {})
        assert result["overview"] == "Short sync."
        assert result["key_points"] == []
        assert result["decisions"] == []
        assert result["action_items"] == []
        assert result["open_questions"] == []
        assert result["facts"] == []


def test_completion_raises_empty_completion_error_on_whitespace_or_length_finish() -> None:
    mock_client = MagicMock()

    # Empty content case
    mock_choice_empty = MagicMock(finish_reason="stop")
    mock_choice_empty.message.content = "   "
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice_empty])

    with pytest.raises(EmptyCompletionError):
        _completion(mock_client, "sys_prompt", "user_prompt", "model", {})

    # Truncated length finish_reason case
    mock_choice_length = MagicMock(finish_reason="length")
    mock_choice_length.message.content = '{"partial": true'
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice_length])

    with pytest.raises(EmptyCompletionError):
        _completion(mock_client, "sys_prompt", "user_prompt", "model", {})


def test_summarize_transcript_mocked_pipeline() -> None:
    transcript = "[00:00:01] Alice: We will release next week."

    section_output = {
        "facts": ["Release planned for next week."],
        "decisions": [{"text": "Release next week", "timestamp": "00:00:01"}],
        "action_items": [],
        "open_questions": [],
    }
    final_output = {
        "overview": "The team agreed to release next week.",
        "key_points": ["Release is next week."],
        "decisions": [{"text": "Release next week", "timestamp": "00:00:01"}],
        "action_items": [],
        "open_questions": [],
    }

    with patch("app.services.summarizer.Groq") as mock_groq_cls:
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client

        with patch("app.services.summarizer._json_completion", side_effect=[section_output, final_output]) as mock_json_comp:
            result = summarize_transcript(transcript, api_key="fake-key", model="test-model")

            assert result == final_output
            assert mock_json_comp.call_count == 2


def test_summarize_section_backoff_recursively_splits_on_error() -> None:
    mock_client = MagicMock()
    # Large chunk > MIN_CHUNK_CHARS (2000)
    large_chunk = ("Line of text content\n" * 150)
    assert len(large_chunk) > 2000

    success_side_left = {"facts": ["Left fact"], "decisions": [], "action_items": [], "open_questions": []}
    success_side_right = {"facts": ["Right fact"], "decisions": [], "action_items": [], "open_questions": []}

    # First call on full chunk raises EmptyCompletionError, sub-calls succeed
    with patch("app.services.summarizer._json_completion", side_effect=[EmptyCompletionError("too long"), success_side_left, success_side_right]):
        merged = _summarize_section_with_backoff(mock_client, large_chunk, "test-model")
        assert merged["facts"] == ["Left fact", "Right fact"]

