import json
from typing import Any

from groq import APIConnectionError, InternalServerError, RateLimitError, Groq
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

MAX_CHUNK_CHARS, MIN_CHUNK_CHARS, MAX_COMPLETION_TOKENS = 12000, 2000, 4096

SECTION_SYSTEM_PROMPT = """You summarize one section of a meeting transcript. Use only information explicitly stated in the transcript. Do not infer or invent owners, dates, decisions, or action items. Use null for an explicitly mentioned but unknown owner or deadline.

Return exactly one JSON object with every one of these keys: facts, decisions, action_items, open_questions. Every key is required, even when it has no content. Use [] for an empty list - never omit a key. Each decision must have text and timestamp. Each action item must have task, owner, deadline, and timestamp. Each open question must have text and timestamp."""

DIRECT_SYSTEM_PROMPT = """You summarize a meeting transcript into an action-oriented summary. Use only information explicitly stated in the transcript. Do not infer or invent owners, dates, decisions, or action items. Use null for an explicitly mentioned but unknown owner or deadline.

Return exactly one JSON object with every one of these keys: overview, key_points, decisions, action_items, open_questions. Every key is required, even when it has no content. Use [] for an empty list - never omit a key. Each decision must have text and timestamp. Each action item must have task, owner, deadline, and timestamp. Each open question must have text and timestamp."""

FINAL_SYSTEM_PROMPT = """Combine these factual section summaries into one final meeting summary. Use only supplied facts. Do not create or infer missing details. Deduplicate repeated entries.

Return exactly one JSON object with every one of these keys: overview, key_points, decisions, action_items, open_questions. Every key is required. Use [] for an empty list - never omit a key. Each decision must have text and timestamp. Each action item must have task, owner, deadline, and timestamp. Each open question must have text and timestamp."""

_DECISION = {"type": "object", "properties": {"text": {"type": "string"}, "timestamp": {"type": ["string", "null"]}}, "required": ["text", "timestamp"], "additionalProperties": False}
_ACTION_ITEM = {"type": "object", "properties": {"task": {"type": "string"}, "owner": {"type": ["string", "null"]}, "deadline": {"type": ["string", "null"]}, "timestamp": {"type": ["string", "null"]}}, "required": ["task", "owner", "deadline", "timestamp"], "additionalProperties": False}
_OPEN_QUESTION = {"type": "object", "properties": {"text": {"type": "string"}, "timestamp": {"type": ["string", "null"]}}, "required": ["text", "timestamp"], "additionalProperties": False}
SECTION_SCHEMA = {"name": "section_summary", "strict": False, "schema": {"type": "object", "properties": {"facts": {"type": "array", "items": {"type": "string"}}, "decisions": {"type": "array", "items": _DECISION}, "action_items": {"type": "array", "items": _ACTION_ITEM}, "open_questions": {"type": "array", "items": _OPEN_QUESTION}}, "required": ["facts"], "additionalProperties": False}}
FINAL_SCHEMA = {"name": "final_summary", "strict": False, "schema": {"type": "object", "properties": {"overview": {"type": "string"}, "key_points": {"type": "array", "items": {"type": "string"}}, "decisions": {"type": "array", "items": _DECISION}, "action_items": {"type": "array", "items": _ACTION_ITEM}, "open_questions": {"type": "array", "items": _OPEN_QUESTION}}, "required": ["overview"], "additionalProperties": False}}


class EmptyCompletionError(Exception):
    pass


def chunk_transcript(transcript: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    if len(transcript) <= max_chars:
        return [transcript]
    chunks, current = [], ""
    for line in transcript.splitlines():
        if current and len(current) + len(line) + 1 > max_chars:
            chunks.append(current)
            current = ""
        current += line + "\n"
    if current:
        chunks.append(current)
    return chunks


@retry(retry=retry_if_exception_type((APIConnectionError, InternalServerError, RateLimitError)), wait=wait_exponential(min=1, max=8), stop=stop_after_attempt(3), reraise=True)
def _completion(client: Groq, system_prompt: str, user_prompt: str, model: str, schema: dict[str, Any]) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        reasoning_effort="low",
        response_format={"type": "json_schema", "json_schema": schema},
    )
    choice, content = response.choices[0], response.choices[0].message.content or ""
    if not content.strip() or choice.finish_reason == "length":
        raise EmptyCompletionError(f"No complete JSON (finish_reason={choice.finish_reason!r}).")
    return content


def _json_completion(client: Groq, system_prompt: str, user_prompt: str, model: str, schema: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(_completion(client, system_prompt, user_prompt, model, schema))
    # Models occasionally omit empty arrays. Keep the stored/API contract stable.
    for key in ("key_points", "decisions", "action_items", "open_questions"):
        result.setdefault(key, [])
    result.setdefault("facts", [])
    return result


def _split_for_backoff(chunk: str) -> tuple[str, str]:
    midpoint = len(chunk) // 2
    split_at = chunk.rfind("\n", 0, midpoint)
    if split_at < MIN_CHUNK_CHARS // 2:
        split_at = midpoint
    else:
        split_at += 1
    return chunk[:split_at], chunk[split_at:]


def _summarize_section_with_backoff(client: Groq, chunk: str, model: str) -> dict[str, Any]:
    try:
        user_content = f"Transcript section:\n{chunk}"
        return _json_completion(client, SECTION_SYSTEM_PROMPT, user_content, model, SECTION_SCHEMA)
    except EmptyCompletionError:
        if len(chunk) <= MIN_CHUNK_CHARS:
            raise
        left, right = _split_for_backoff(chunk)
        left_result = _summarize_section_with_backoff(client, left, model)
        right_result = _summarize_section_with_backoff(client, right, model)
        return {"facts": left_result["facts"] + right_result["facts"], "decisions": left_result["decisions"] + right_result["decisions"], "action_items": left_result["action_items"] + right_result["action_items"], "open_questions": left_result["open_questions"] + right_result["open_questions"]}


def summarize_transcript(transcript: str, api_key: str, model: str = "openai/gpt-oss-20b") -> dict[str, Any]:
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured.")
    client = Groq(api_key=api_key)
    chunks = chunk_transcript(transcript)

    # For single-chunk transcripts, summarize directly in 1 LLM call to save latency and cost
    if len(chunks) == 1:
        try:
            user_content = f"Meeting transcript:\n{chunks[0]}"
            return _json_completion(client, DIRECT_SYSTEM_PROMPT, user_content, model, FINAL_SCHEMA)
        except EmptyCompletionError:
            # Fall back to Map-Reduce if the single chunk encounters a completion overflow
            pass

    section_summaries = [_summarize_section_with_backoff(client, chunk, model) for chunk in chunks]
    final_user_content = f"Section summaries:\n{json.dumps(section_summaries)}"
    return _json_completion(client, FINAL_SYSTEM_PROMPT, final_user_content, model, FINAL_SCHEMA)
