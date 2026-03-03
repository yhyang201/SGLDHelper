---
name: moonshot-kimi-api
description: "Guide for building applications with the Moonshot Kimi API, focusing on the Kimi K2.5 model. Use this skill whenever the user wants to integrate or call the Kimi API, use Moonshot's large language models (kimi-k2.5, kimi-k2, kimi-k2-thinking), implement thinking/instant modes, tool calling / function calling with Kimi, streaming output, JSON mode, partial mode (prefix continuation), web search, multimodal input (images/video), or multi-turn conversations via Kimi API. Also trigger when code imports openai and sets base_url to api.moonshot.ai or api.moonshot.cn, or when the user mentions 'moonshot', 'kimi', 'kimi-k2', or 'kimi-k2.5' in an API context."
---

# Moonshot Kimi API Development Guide

Build applications powered by Moonshot's Kimi large language models. The API is OpenAI-compatible — use the standard `openai` Python/Node SDK with a different `base_url`.

## Quick Reference

| Item | Value |
|------|-------|
| Base URL (Global) | `https://api.moonshot.ai/v1` |
| Base URL (China) | `https://api.moonshot.cn/v1` |
| Auth Header | `Authorization: Bearer $MOONSHOT_API_KEY` |
| Primary Model | `kimi-k2.5` (multimodal, 256K context, 1T MoE / 32B active) |
| Other Models | `kimi-k2`, `kimi-k2-0905`, `kimi-k2-thinking` |
| Chat Endpoint | `POST /v1/chat/completions` |
| Files Endpoint | `POST /v1/files` |
| Pricing | ~$0.60/M input, ~$2.50/M output tokens |

## Reference Files

Read these for detailed examples when implementing specific features:

| File | Contents |
|------|----------|
| `references/tool-calling.md` | Tool/function calling definitions, execution loop, streaming tool calls, multi-step agentic patterns |
| `references/streaming.md` | Basic streaming, thinking-mode streaming, tool-call streaming, auto-reconnect |
| `references/advanced-features.md` | JSON mode, JSON schema, partial mode, web search, multimodal (image/video), file API, error handling, migration guide, complete agent example |

---

## Setup & Authentication

Get an API key from [platform.moonshot.ai](https://platform.moonshot.ai/).

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-moonshot-api-key",  # or os.environ["MOONSHOT_API_KEY"]
    base_url="https://api.moonshot.ai/v1",
)
```

```javascript
import OpenAI from "openai";
const client = new OpenAI({
  apiKey: "your-moonshot-api-key",
  baseURL: "https://api.moonshot.ai/v1",
});
```

---

## Models

**Kimi K2.5** (recommended default): Model ID `kimi-k2.5`. 1T-parameter MoE (32B active). 256K context. Native multimodal (text + image + video), agentic tool calling, thinking & instant modes, web search. Vision encoder: MoonViT (400M params).

**Kimi K2**: Model ID `kimi-k2` or `kimi-k2-0905`. 128K/256K context. Text-only, strong tool calling.

**Kimi K2 Thinking**: Model ID `kimi-k2-thinking`. 256K context. Dedicated reasoning model with interleaved thinking and multi-step tool calling.

---

## Basic Chat Completion

```python
response = client.chat.completions.create(
    model="kimi-k2.5",
    messages=[
        {"role": "system", "content": "You are Kimi, an AI assistant created by Moonshot AI."},
        {"role": "user", "content": "Explain quantum computing in simple terms."},
    ],
    temperature=0.6,
    top_p=0.95,
    max_tokens=4096,
)
print(response.choices[0].message.content)
```

### Key Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | string | Model ID (e.g. `"kimi-k2.5"`) |
| `messages` | array | Conversation messages with `role` and `content` |
| `temperature` | float | 0.0–1.0. Use **0.6** for instant, **1.0** for thinking |
| `top_p` | float | Nucleus sampling, recommended **0.95** |
| `max_tokens` | int | Max output tokens (e.g. 4096, 8192) |
| `stream` | bool | Enable streaming (default: false) |
| `stop` | array | Up to 4 stop sequences |
| `n` | int | Number of completions to generate |
| `response_format` | object | `{"type": "json_object"}` or `{"type": "json_schema", ...}` |
| `tools` | array | Function definitions for tool calling |
| `tool_choice` | string/object | `"auto"` / `"none"` / `"required"` / specific function |
| `frequency_penalty` | float | Penalize frequent tokens |
| `presence_penalty` | float | Penalize repeated tokens |

---

## Thinking Mode vs. Instant Mode

Kimi K2.5 supports two modes controlling whether the model shows internal reasoning.

### Thinking Mode (Default)

Produces reasoning traces before answering. Best for math, logic, code debugging, research. Response includes `reasoning_content` (the thinking process) and `content` (the final answer).

```python
response = client.chat.completions.create(
    model="kimi-k2.5",
    messages=[{"role": "user", "content": "Prove that sqrt(2) is irrational."}],
    temperature=1.0,   # recommended for thinking
    top_p=0.95,
    max_tokens=8192,
)
reasoning = response.choices[0].message.reasoning_content  # thinking trace
answer = response.choices[0].message.content                # final answer
```

Thinking budget tiers: ~8K tokens (routine), ~32K (complex), ~96K (frontier).

### Instant Mode (Disable Thinking)

Skips reasoning traces — faster and 60–75% cheaper on tokens. Use for translation, summarization, simple Q&A, creative writing.

```python
response = client.chat.completions.create(
    model="kimi-k2.5",
    messages=[{"role": "user", "content": "Translate 'hello world' to French."}],
    temperature=0.6,   # recommended for instant
    top_p=0.95,
    max_tokens=4096,
    extra_body={"thinking": {"type": "disabled"}},
)
```

Alternative syntax for vLLM/SGLang self-hosted deployments:
```python
extra_body={"chat_template_kwargs": {"thinking": False}}
```

| Use Case | Mode | Temperature |
|----------|------|-------------|
| Math / Logic / Proofs | Thinking | 1.0 |
| Code debugging & analysis | Thinking | 1.0 |
| Complex research | Thinking | 1.0 |
| Translation / Summarization | Instant | 0.6 |
| Simple Q&A / Creative writing | Instant | 0.6 |
| Tool-calling agents | Either | 0.6–1.0 |

---

## Tool Calling (Overview)

Kimi K2.5 has strong native tool-calling. Define tools as functions, pass them in requests, and the model decides when and how to invoke them. The model can perform **interleaved thinking and multi-step tool calling** — reasoning between tool calls, handling 200+ sequential calls in agentic workflows.

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "parameters": {
            "type": "object",
            "required": ["city"],
            "properties": {
                "city": {"type": "string", "description": "City name"}
            }
        }
    }
}]

response = client.chat.completions.create(
    model="kimi-k2.5",
    messages=messages,
    tools=tools,
    tool_choice="auto",
    temperature=0.6,
)
```

When `finish_reason == "tool_calls"`, execute the requested functions and feed results back as `role: "tool"` messages. See `references/tool-calling.md` for the complete execution loop, streaming tool calls, and `tool_choice` options.

---

## JSON Mode (Overview)

Force valid JSON output with `response_format`. Always include a JSON instruction in your system/user message.

```python
response = client.chat.completions.create(
    model="kimi-k2.5",
    messages=[
        {"role": "system", "content": "Respond in JSON format."},
        {"role": "user", "content": "Extract name, age from: 'John is a 30-year-old engineer.'"},
    ],
    response_format={"type": "json_object"},
    temperature=0.6,
)
```

For strict structure, use `{"type": "json_schema", "json_schema": {...}}`. See `references/advanced-features.md` for JSON schema examples.

---

## Partial Mode (Overview)

Provide a prefix the model must continue from. Set the last message to `role: "assistant"` with the prefix and add `extra_body={"partial": True}`.

```python
response = client.chat.completions.create(
    model="kimi-k2.5",
    messages=[
        {"role": "user", "content": "Write a fibonacci function."},
        {"role": "assistant", "content": "def fibonacci(n):\n    "},
    ],
    extra_body={"partial": True},
    temperature=0.6,
)
```

Useful for code completion, structured output prefixes, constrained generation. See `references/advanced-features.md`.

---

## Streaming (Overview)

Enable token-by-token output with `stream=True`. In thinking mode, `reasoning_content` appears in delta chunks before `content` chunks.

```python
stream = client.chat.completions.create(
    model="kimi-k2.5",
    messages=[{"role": "user", "content": "Write a short story."}],
    stream=True,
    temperature=0.6,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

See `references/streaming.md` for thinking-mode streaming, tool-call streaming, and auto-reconnect patterns.

---

## Web Search (Overview)

Enable built-in internet search with the `$web_search` tool (~$0.005/call). The model autonomously decides when to search.

```python
response = client.chat.completions.create(
    model="kimi-k2.5",
    messages=[{"role": "user", "content": "Latest quantum computing news?"}],
    tools=[{"type": "builtin_function", "function": {"name": "$web_search"}}],
    tool_choice="auto",
    temperature=0.6,
)
```

---

## Multimodal Input (Overview)

Kimi K2.5 natively processes images and video. Use `image_url` or `video_url` content types in messages.

```python
response = client.chat.completions.create(
    model="kimi-k2.5",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}},
        ],
    }],
    temperature=0.6,
)
```

Supports URL and base64 for images (JPG/PNG/GIF/WEBP) and video (MP4). Kimi K2.5 excels at converting wireframes/mockups into code. See `references/advanced-features.md` for video input and base64 examples.

---

## Multi-Turn Conversations

Accumulate messages across turns. The 256K context window handles long conversations, but implement sliding window or summarization for very long sessions.

```python
messages = [{"role": "system", "content": "You are Kimi, a helpful assistant."}]

def chat(user_input: str) -> str:
    messages.append({"role": "user", "content": user_input})
    response = client.chat.completions.create(
        model="kimi-k2.5", messages=messages, temperature=0.6, max_tokens=4096,
    )
    reply = response.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})
    return reply
```

---

## Best Practices

1. **Choose the right mode**: Instant for simple tasks (60–75% token savings), thinking only when reasoning quality matters
2. **Temperature**: 0.6 instant, 1.0 thinking, top_p 0.95 for both
3. **Streaming** for user-facing apps to reduce perceived latency
4. **Retries** with exponential backoff for production (see `references/advanced-features.md`)
5. **Monitor tokens** via the `usage` field in responses
6. **JSON mode** for structured extraction instead of parsing free text
7. **Partial mode** for code completion and constrained generation

---

## Migration from OpenAI

Minimal changes — swap `base_url` and `model`:

```python
# OpenAI → Kimi
client = OpenAI(api_key="moonshot-key", base_url="https://api.moonshot.ai/v1")
response = client.chat.completions.create(model="kimi-k2.5", ...)
```

Key differences: temperature range 0–1 (not 0–2), `reasoning_content` in thinking mode, `extra_body` for instant/partial modes, `$web_search` built-in tool, native video input. See `references/advanced-features.md` for full migration notes.
