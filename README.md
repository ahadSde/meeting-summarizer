# Meeting Summarizer

An asynchronous, local-first meeting-audio pipeline. Upload a recording and receive a timestamped transcript plus an action-oriented summary with decisions, action items, and open questions.

## Demo Video

[Watch the demo](https://github.com/user-attachments/assets/9a1e4282-332a-43c2-955a-70443a365713)

## Architecture

```text
Browser (HTML + Tailwind)
        | POST /api/meetings
        v
FastAPI ingestion -> SQLite (meeting metadata, transcript, summary)
        | background task
        v
ffmpeg media validation / 16 kHz mono normalization
        v
local faster-whisper native transcription (Silero VAD)
        v
Groq transcript summarization (Direct / Map-Reduce)
        v
GET /api/meetings/{id} -> results UI
```

## Stack

| Concern | Choice | Rationale |
| --- | --- | --- |
| API | FastAPI | Typed endpoints and a clear background-task workflow |
| Audio | ffmpeg | Media inspection, validation, and mono 16 kHz normalization |
| ASR | local `faster-whisper` | Native 30s sliding-window transcription with Silero VAD; data stays local |
| LLM | Groq `openai/gpt-oss-20b` | Fast summary generation with structured output support |
| Storage | SQLite + SQLAlchemy | Durable results with minimal setup |
| UI | HTML, JavaScript, Tailwind CDN | Fast, dependency-light demo interface |

## Pipeline and reliability choices

1. The upload endpoint checks format, size, readable duration, and maximum duration, then creates a persisted `queued` meeting.
2. A background job normalizes audio to mono 16 kHz 16-bit PCM WAV matching Whisper's native acoustic model requirements.
3. `faster-whisper` transcribes the audio stream natively with Silero VAD silence filtering, avoiding boundary clipping.
4. Short transcripts are summarized in a single direct pass; long transcripts are split into sections with Map-Reduce combining.
5. Prompts require information to be explicitly supported by the transcript. Unknown owner/deadline fields are `null`.
6. Groq is asked for JSON-schema output. The app normalizes missing empty categories to `[]`, retries only transient failures (network, server, rate limit), and records a safe error on failure.
7. The browser polls the result endpoint without holding the upload request open. Temporary audio files are removed after processing.

## Example result

```json
{
  "overview": "The team reviewed the release plan and outstanding testing work.",
  "key_points": ["The release target is Friday."],
  "decisions": [{ "text": "Ship version 2.1 on Friday.", "timestamp": "00:14:32" }],
  "action_items": [{
    "task": "Prepare release notes",
    "owner": "Dev",
    "deadline": "Thursday",
    "timestamp": "00:18:05"
  }],
  "open_questions": []
}
```

## Setup

### Prerequisites

- Python 3.9+
- [ffmpeg](https://ffmpeg.org/)
- A Groq API key

```bash
git clone <your-repository-url>
cd meeting-summarizer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set `GROQ_API_KEY` in `.env`, then run:

```bash
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`. The first transcription downloads the selected Whisper model.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `GROQ_API_KEY` | required | Used only for summaries |
| `GROQ_MODEL` | `openai/gpt-oss-20b` | Groq chat model |
| `WHISPER_MODEL` | `base` | Try `small` for higher accuracy at slower speed |
| `MAX_UPLOAD_SIZE_MB` | `100` | Upload limit |
| `MAX_AUDIO_DURATION_MINUTES` | `120` | Meeting duration limit |

## API

| Endpoint | Description |
| --- | --- |
| `POST /api/meetings` | Multipart upload field: `audio`; returns queued meeting metadata |
| `GET /api/meetings/{id}` | Returns job status, transcript, summary, timings, and safe errors |
| `GET /api/health` | Service/configuration health check |

## Tests

```bash
pytest -q
```

The suite covers API endpoints, file streaming validation, audio normalization and transcription, ASR transcription caching, Map-Reduce summarization, backoff error recovery, and background job lifecycle.

## Known limitations

- No speaker diarization yet; transcript segments are timestamped but not speaker-labeled.
- Whisper accuracy/speed depend on hardware and model size.
- Groq free-tier limits can delay long meetings; retries use exponential backoff.
- In-process background jobs are right for a demo; production should use a durable queue and object storage.
- Meeting history and Markdown/JSON export are sensible next UI additions.

## Security

- Never commit `.env` or API keys; `.env` is Git-ignored.
- Temporary uploads are removed after processing.
- A production deployment would add authentication, encrypted storage, retention controls, and recording-consent notices.
