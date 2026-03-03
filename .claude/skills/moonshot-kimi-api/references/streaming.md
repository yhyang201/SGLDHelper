# Streaming — Detailed Guide

## Table of Contents
1. [Basic Streaming](#basic-streaming)
2. [Streaming with Thinking Mode](#streaming-with-thinking-mode)
3. [Streaming with Tool Calls](#streaming-with-tool-calls)
4. [Auto-Reconnect Pattern](#auto-reconnect-pattern)

---

## Basic Streaming

Enable token-by-token output by setting `stream=True`. Responses arrive as server-sent events (SSE) with delta content.

```python
stream = client.chat.completions.create(
    model="kimi-k2.5",
    messages=[
        {"role": "user", "content": "Write a short story about a robot."},
    ],
    temperature=0.6,
    max_tokens=4096,
    stream=True,
)

for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.content:
        print(delta.content, end="", flush=True)
print()  # newline after streaming completes
```

The final chunk includes `finish_reason` (e.g. `"stop"`) and optionally a `usage` field with token counts.

---

## Streaming with Thinking Mode

In thinking mode, the response has two phases:
1. **Reasoning phase**: `reasoning_content` appears in delta chunks (the model's internal thinking)
2. **Answer phase**: `content` appears in delta chunks (the final response)

```python
stream = client.chat.completions.create(
    model="kimi-k2.5",
    messages=[
        {"role": "user", "content": "Solve: what is 1234 * 5678?"},
    ],
    temperature=1.0,
    stream=True,
)

reasoning_text = ""
answer_text = ""

for chunk in stream:
    delta = chunk.choices[0].delta

    # Phase 1: Reasoning trace chunks
    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
        reasoning_text += delta.reasoning_content
        print(f"[Thinking] {delta.reasoning_content}", end="", flush=True)

    # Phase 2: Final answer chunks
    if delta.content:
        answer_text += delta.content
        print(delta.content, end="", flush=True)

print(f"\n\nFull reasoning: {reasoning_text}")
print(f"Final answer: {answer_text}")
```

---

## Streaming with Tool Calls

See `tool-calling.md` → "Streaming Tool Calls" section for the complete pattern. Tool call deltas arrive incrementally and must be assembled from partial chunks.

---

## Auto-Reconnect Pattern

For long-running streams, network interruptions can occur. Implement retry logic:

```python
import time
from openai import APIConnectionError, APITimeoutError

def stream_with_retry(client, max_retries=3, **kwargs):
    """Stream with automatic retry on connection failure."""
    kwargs["stream"] = True
    collected_content = ""

    for attempt in range(max_retries):
        try:
            stream = client.chat.completions.create(**kwargs)
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    collected_content += delta.content
                    yield delta.content

                # Track usage from final chunk
                if hasattr(chunk, "usage") and chunk.usage:
                    yield {"usage": chunk.usage}
            return  # Success — exit retry loop

        except (APIConnectionError, APITimeoutError) as e:
            if attempt < max_retries - 1:
                delay = 2 ** attempt
                print(f"\nConnection lost. Reconnecting in {delay}s...")
                time.sleep(delay)
                # On retry, you may need to adjust messages to account for
                # partial content already received
            else:
                raise

# Usage
for token in stream_with_retry(
    client,
    model="kimi-k2.5",
    messages=[{"role": "user", "content": "Write a long essay."}],
    temperature=0.6,
    max_tokens=8192,
):
    if isinstance(token, str):
        print(token, end="", flush=True)
    elif isinstance(token, dict) and "usage" in token:
        print(f"\nTokens used: {token['usage']}")
```

**Tips:**
- Use the `usage` field in the final chunk to track token consumption for billing
- For very long generations, consider checkpointing partial content
- The OpenAI SDK has built-in retry logic for transient errors, but streaming reconnection needs manual handling
