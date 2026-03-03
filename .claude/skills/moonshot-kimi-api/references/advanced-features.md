# Advanced Features — Detailed Guide

## Table of Contents
1. [JSON Mode](#json-mode)
2. [JSON Schema Mode](#json-schema-mode)
3. [Partial Mode (Prefix Continuation)](#partial-mode)
4. [Web Search](#web-search)
5. [Multimodal Input (Images & Video)](#multimodal-input)
6. [File API](#file-api)
7. [Multi-Turn Conversations](#multi-turn-conversations)
8. [Error Handling & Retry](#error-handling--retry)
9. [Migration from OpenAI](#migration-from-openai)

---

## JSON Mode

Force the model to output valid JSON. Always include a JSON instruction in the system or user message — the `response_format` parameter constrains the format but the model needs the instruction to know *what* JSON to produce.

### Basic JSON Object

```python
response = client.chat.completions.create(
    model="kimi-k2.5",
    messages=[
        {"role": "system", "content": "You are a helpful assistant. Always respond in JSON format."},
        {"role": "user", "content": "Extract the name, age, and occupation from: 'John is a 30-year-old engineer.'"},
    ],
    temperature=0.6,
    response_format={"type": "json_object"},
)

import json
data = json.loads(response.choices[0].message.content)
print(data)
# {"name": "John", "age": 30, "occupation": "engineer"}
```

---

## JSON Schema Mode

For stricter output structure, provide a full JSON schema. The model's output will conform to the schema.

```python
response = client.chat.completions.create(
    model="kimi-k2.5",
    messages=[
        {"role": "user", "content": "List 3 programming languages with their year of creation."},
    ],
    temperature=0.6,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "languages",
            "schema": {
                "type": "object",
                "properties": {
                    "languages": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "year": {"type": "integer"},
                                "paradigm": {"type": "string"}
                            },
                            "required": ["name", "year"]
                        }
                    }
                },
                "required": ["languages"]
            }
        }
    },
)
```

**When to use which:**
- `json_object`: When you need valid JSON but the structure is flexible
- `json_schema`: When you need a specific, guaranteed structure (e.g., feeding into a typed API)

---

## Partial Mode

Partial mode (prefix continuation) lets you provide an assistant message prefix that the model must continue from. The model's response begins exactly where your prefix left off.

### Setup

Set the **last message** to `role: "assistant"` with your prefix, and pass `extra_body={"partial": True}`.

```python
response = client.chat.completions.create(
    model="kimi-k2.5",
    messages=[
        {"role": "system", "content": "You are a helpful code assistant."},
        {"role": "user", "content": "Write a Python function to calculate fibonacci numbers."},
        {"role": "assistant", "content": "def fibonacci(n):\n    "},  # prefix
    ],
    temperature=0.6,
    extra_body={"partial": True},
)

# Concatenate prefix + completion
full_code = "def fibonacci(n):\n    " + response.choices[0].message.content
print(full_code)
```

### Use Cases

- **Code completion**: Provide function signature → model fills the body
- **Structured output prefix**: Start with `{"result": [` → model completes the JSON
- **Translation with format**: Provide target format prefix → model continues
- **Constrained generation**: Start a sentence → model finishes in the same style
- **Fill-in-the-middle**: Provide surrounding context → model fills the gap

### Partial Mode + JSON

Combine partial mode with JSON for highly controlled output:

```python
response = client.chat.completions.create(
    model="kimi-k2.5",
    messages=[
        {"role": "user", "content": "List 3 fruits with colors."},
        {"role": "assistant", "content": '{"fruits": ['},
    ],
    extra_body={"partial": True},
    temperature=0.6,
)
# Complete JSON: '{"fruits": [' + response content
```

---

## Web Search

Kimi API provides built-in internet search via the `$web_search` tool. The model autonomously decides when to search.

```python
response = client.chat.completions.create(
    model="kimi-k2.5",
    messages=[
        {"role": "system", "content": "You are a helpful assistant with internet access."},
        {"role": "user", "content": "What are the latest developments in quantum computing this week?"},
    ],
    tools=[
        {
            "type": "builtin_function",
            "function": {
                "name": "$web_search",
            }
        }
    ],
    tool_choice="auto",
    temperature=0.6,
)
print(response.choices[0].message.content)
```

### Combining Web Search with Custom Tools

You can pass both `$web_search` and your own function tools in the same `tools` array:

```python
tools = [
    {"type": "builtin_function", "function": {"name": "$web_search"}},
    {
        "type": "function",
        "function": {
            "name": "save_to_database",
            "description": "Save extracted information to the database.",
            "parameters": {
                "type": "object",
                "required": ["data"],
                "properties": {"data": {"type": "object"}}
            }
        }
    },
]
```

**Cost**: ~$0.005 per search invocation.

---

## Multimodal Input

Kimi K2.5 natively processes images and video alongside text.

### Image from URL

```python
response = client.chat.completions.create(
    model="kimi-k2.5",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What is shown in this image?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}},
        ],
    }],
    temperature=0.6,
)
```

### Image from Base64

```python
import base64

with open("screenshot.png", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

response = client.chat.completions.create(
    model="kimi-k2.5",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this screenshot in detail."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
        ],
    }],
    temperature=0.6,
)
```

Supported image formats: JPG, JPEG, PNG, GIF, WEBP.

### Video Input

```python
import base64

with open("clip.mp4", "rb") as f:
    vid_b64 = base64.b64encode(f.read()).decode()

response = client.chat.completions.create(
    model="kimi-k2.5",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe what happens in this video."},
            {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{vid_b64}"}},
        ],
    }],
    temperature=0.6,
)
```

### Visual Coding

Kimi K2.5 excels at converting visual designs to code. Pass wireframes, UI mockups, or screenshots and ask for React/HTML/CSS:

```python
response = client.chat.completions.create(
    model="kimi-k2.5",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Convert this wireframe to a React component with Tailwind CSS."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{wireframe_b64}"}},
        ],
    }],
    temperature=0.6,
    max_tokens=8192,
)
```

---

## File API

Upload documents for the model to reference in conversations. Useful for long PDFs, codebases, or data files.

### Upload

```python
file = client.files.create(
    file=open("document.pdf", "rb"),
    purpose="file-extract",
)
print(file.id)  # e.g. "file-abc123"
```

### Reference in Conversation

Pass the file ID as a system message:

```python
response = client.chat.completions.create(
    model="kimi-k2.5",
    messages=[
        {"role": "system", "content": "You are a document analysis assistant."},
        {"role": "system", "content": file.id},
        {"role": "user", "content": "Summarize the key findings in this document."},
    ],
    temperature=0.6,
)
```

### Manage Files

```python
# List all uploaded files
files = client.files.list()

# Get file metadata
info = client.files.retrieve("file-abc123")

# Delete a file
client.files.delete("file-abc123")
```

---

## Multi-Turn Conversations

### Basic Pattern

```python
messages = [{"role": "system", "content": "You are Kimi, a helpful AI assistant."}]

def chat(user_input: str) -> str:
    messages.append({"role": "user", "content": user_input})
    response = client.chat.completions.create(
        model="kimi-k2.5",
        messages=messages,
        temperature=0.6,
        max_tokens=4096,
    )
    reply = response.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})
    return reply

print(chat("What is Python?"))
print(chat("What are its main advantages?"))   # context preserved
print(chat("Show me a code example."))          # still has full context
```

### Context Management for Long Conversations

The 256K context window is large but not infinite. For very long sessions:

```python
import tiktoken

def trim_messages(messages, max_tokens=200000):
    """Keep system message + most recent messages within token limit."""
    encoder = tiktoken.encoding_for_model("gpt-4")  # approximate
    system_msgs = [m for m in messages if m["role"] == "system"]
    other_msgs = [m for m in messages if m["role"] != "system"]

    total = sum(len(encoder.encode(str(m["content"]))) for m in system_msgs)

    kept = []
    for msg in reversed(other_msgs):
        msg_tokens = len(encoder.encode(str(msg["content"])))
        if total + msg_tokens > max_tokens:
            break
        kept.insert(0, msg)
        total += msg_tokens

    return system_msgs + kept
```

---

## Error Handling & Retry

### Common Error Codes

| Code | Meaning | Action |
|------|---------|--------|
| 400 | Bad request | Check parameters |
| 401 | Invalid API key | Verify key |
| 403 | Forbidden | Check permissions |
| 404 | Not found | Check endpoint |
| 429 | Rate limit | Backoff + retry |
| 500 | Server error | Backoff + retry |

### Rate Limits (by account balance)

| Balance | Rate |
|---------|------|
| 1–100 credits | 20 req/min |
| 101–1,000 | 60 req/min |
| 1,001–10,000 | 200 req/min |
| 10,000+ | 500 req/min |

### Exponential Backoff

```python
import time
from openai import RateLimitError, APIError

def call_with_retry(func, max_retries=3, base_delay=1.0):
    for attempt in range(max_retries):
        try:
            return func()
        except RateLimitError:
            delay = base_delay * (2 ** attempt)
            print(f"Rate limited. Retrying in {delay}s...")
            time.sleep(delay)
        except APIError as e:
            if e.status_code >= 500:
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
            else:
                raise
    raise Exception("Max retries exceeded")
```

---

## Migration from OpenAI

### Minimal Changes

```python
# Before (OpenAI)
client = OpenAI(api_key="sk-...")
response = client.chat.completions.create(model="gpt-4o", ...)

# After (Kimi)
client = OpenAI(
    api_key="your-moonshot-key",
    base_url="https://api.moonshot.ai/v1",
)
response = client.chat.completions.create(model="kimi-k2.5", ...)
```

### Key Differences from OpenAI

| Feature | OpenAI | Kimi |
|---------|--------|------|
| Temperature range | 0–2 | 0–1 |
| Thinking/reasoning | o1/o3 models only | All K2.5 models, controlled via `extra_body` |
| Disable thinking | Not applicable | `extra_body={"thinking": {"type": "disabled"}}` |
| Partial/prefix mode | Not available | `extra_body={"partial": True}` |
| Built-in web search | Not available | `$web_search` tool |
| Video input | Not available | `video_url` content type |
| Reasoning in response | `reasoning` field | `reasoning_content` field |
| Anthropic-compatible endpoint | No | Yes — `/v1/messages` also available |

### Debugging

Use [MoonPalace](https://platform.moonshot.ai/docs/guide/use-moonpalace) to intercept and inspect API requests/responses, including full payloads, latency, and token usage.
