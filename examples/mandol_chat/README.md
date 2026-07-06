# Mandol Chat Demo

`examples/mandol_chat/` is a small runnable demo for the Mandol default real-time memory path.

It demonstrates:

- normal chat messages being inserted as Mandol memory units;
- default automatic session splitting for user messages without an explicit `session_id`;
- memory search over previous chat messages;
- high-level memory generation as an explicit action, not part of every chat insert.

The demo intentionally does not touch LoCoMo / LoCoMo10 / LongMemEval benchmark code.

## Install

From the repository root:

```bash
uv sync
```

Or install only the example dependencies into your current environment:

```bash
pip install -r examples/mandol_chat/requirements.txt
```

## Run

From the repository root:

```bash
python examples/mandol_chat/run.py
```

Then open:

```text
http://127.0.0.1:8008
```

Equivalent uvicorn command:

```bash
PYTHONPATH="$PWD:$PWD/examples/mandol_chat" \
uvicorn mandol_chat.main:app --app-dir examples/mandol_chat --host 127.0.0.1 --port 8008
```

## Environment

Copy `.env.example` if you want to customize settings:

```bash
cp examples/mandol_chat/.env.example examples/mandol_chat/.env
```

Useful variables:

- `MANDOL_CHAT_MOCK_LLM=true`: default. No API key required.
- `MANDOL_CHAT_MOCK_LLM=false`: try to use `mandol.llm.llm_client.LLMClient`.
- `MANDOL_CHAT_LLM_MODEL=gpt-4o-mini-closeai`: model name for real LLM mode.
- `MANDOL_CHAT_REAL_EMBEDDING=false`: default. Avoids loading large embedding models.
- `MANDOL_CHAT_REAL_EMBEDDING=true`: try real `SemanticGraph` / `SemanticMap` vector search.
- `MANDOL_CHAT_DATA_DIR=examples/mandol_chat/mandol_chat/data`: demo data directory.

## How To Use

1. Send a few related messages, such as:
   - `我喜欢软笔书法，也考过七级`
   - `书法练习我想继续记录一下`
2. Watch the current `session_id` remain stable.
3. Send a later or different-topic message in real use and watch auto session metadata change.
4. Use the memory search panel to search `书法`.
5. Click **Finalize Current** to mark the active session finalized.
6. Click **Build High-Level** to explicitly trigger high-level memory build behavior.

In mock mode, high-level build returns a safe mock result. It does not run the expensive L1 / episodic / entity-relation builders.

## Design Notes

- `mandol_chat/services/mandol_service.py` is the only Mandol integration layer.
- Each user and assistant message becomes a `MemoryUnit`.
- User messages without a session use `AutoSessionAssigner`.
- Assistant replies use the same session as the triggering user message.
- Chat inserts are lightweight and synchronous.
- High-level memory build is only triggered by `POST /api/memory/build` or the UI button.
- Benchmark styles (`locomo`, `locomo10`, `longmemeval`) are not imported or modified by this example.

## API

- `GET /api/health`
- `POST /api/chat`
- `POST /api/memory/search`
- `GET /api/sessions`
- `POST /api/sessions/{session_id}/finalize`
- `POST /api/memory/build`
- `POST /api/reset`

## Tests

```bash
python -m pytest examples/mandol_chat/tests -q
```

The tests force mock LLM and mock embedding mode. They do not require external API keys or embedding downloads.

## FAQ

### I do not have an LLM key.

Keep `MANDOL_CHAT_MOCK_LLM=true`. The demo will generate simple context-aware mock replies.

### Embedding model loading is slow.

Keep `MANDOL_CHAT_REAL_EMBEDDING=false`. The demo uses deterministic keyword search so it starts quickly.

### High-level memory build failed or returned mock status.

That is expected in mock mode. Configure a real LLM before running high-level memory builders.

### How do I clear demo data?

Click **Reset** in the UI or call:

```bash
curl -X POST http://127.0.0.1:8008/api/reset
```

