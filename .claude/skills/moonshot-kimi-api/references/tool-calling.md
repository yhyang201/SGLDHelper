# Tool Calling / Function Calling — Detailed Guide

## Table of Contents
1. [Defining Tools](#defining-tools)
2. [Tool Call Execution Loop](#tool-call-execution-loop)
3. [tool_choice Options](#tool_choice-options)
4. [Streaming Tool Calls](#streaming-tool-calls)
5. [Multi-Step Agentic Patterns](#multi-step-agentic-patterns)

---

## Defining Tools

Tools are defined as JSON schemas passed in the `tools` parameter. Each tool has a name, description, and parameter schema.

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city. Call when the user asks about weather.",
            "parameters": {
                "type": "object",
                "required": ["city"],
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name, e.g. 'Beijing'"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search the internal knowledge base for relevant documents.",
            "parameters": {
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 5
                    }
                }
            }
        }
    }
]
```

Write clear descriptions — the model uses them to decide when to call each tool.

---

## Tool Call Execution Loop

The model may request multiple tool calls across multiple turns. Implement a loop that keeps running until the model produces a final text response (`finish_reason != "tool_calls"`).

```python
import json

# Map tool names to actual implementations
def get_weather(city: str) -> dict:
    # Your actual weather API call here
    return {"city": city, "weather": "Sunny", "temp": "25°C"}

def search_knowledge_base(query: str, top_k: int = 5) -> dict:
    # Your actual search implementation here
    return {"results": [{"title": "Result 1", "content": "..."}]}

tool_map = {
    "get_weather": get_weather,
    "search_knowledge_base": search_knowledge_base,
}

messages = [
    {"role": "system", "content": "You are Kimi, a helpful AI assistant."},
    {"role": "user", "content": "What's the weather like in Beijing today?"},
]

finish_reason = None
while finish_reason is None or finish_reason == "tool_calls":
    response = client.chat.completions.create(
        model="kimi-k2.5",
        messages=messages,
        temperature=0.6,
        tools=tools,
        tool_choice="auto",
    )

    choice = response.choices[0]
    finish_reason = choice.finish_reason

    if finish_reason == "tool_calls":
        # Append the assistant message (contains the tool_calls array)
        messages.append(choice.message)

        # Execute each tool call and append the result
        for tool_call in choice.message.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)
            result = tool_map[func_name](**func_args)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": func_name,
                "content": json.dumps(result),
            })

# Final text response
print(choice.message.content)
```

**Key points:**
- Always append `choice.message` (the assistant message with tool calls) before appending tool results
- Each tool result must reference `tool_call_id` to match the call
- The `content` of a tool message must be a string (use `json.dumps` for dict results)

---

## tool_choice Options

| Value | Behavior |
|-------|----------|
| `"auto"` | Model decides whether to call tools (default) |
| `"none"` | Never call tools — text response only |
| `"required"` | Model must call at least one tool |
| `{"type": "function", "function": {"name": "get_weather"}}` | Force a specific function |

---

## Streaming Tool Calls

When streaming with tools, tool call data arrives as incremental deltas. You need to assemble the pieces.

```python
stream = client.chat.completions.create(
    model="kimi-k2.5",
    messages=messages,
    tools=tools,
    tool_choice="auto",
    stream=True,
)

collected_tool_calls = {}

for chunk in stream:
    delta = chunk.choices[0].delta

    # Regular text content
    if delta.content:
        print(delta.content, end="", flush=True)

    # Tool call deltas — assemble incrementally
    if delta.tool_calls:
        for tc in delta.tool_calls:
            idx = tc.index
            if idx not in collected_tool_calls:
                collected_tool_calls[idx] = {
                    "id": "",
                    "function": {"name": "", "arguments": ""},
                }
            if tc.id:
                collected_tool_calls[idx]["id"] = tc.id
            if tc.function and tc.function.name:
                collected_tool_calls[idx]["function"]["name"] += tc.function.name
            if tc.function and tc.function.arguments:
                collected_tool_calls[idx]["function"]["arguments"] += tc.function.arguments

    # When all tool calls are complete
    if chunk.choices[0].finish_reason == "tool_calls":
        for idx, tc_data in sorted(collected_tool_calls.items()):
            func_name = tc_data["function"]["name"]
            func_args = json.loads(tc_data["function"]["arguments"])
            result = tool_map[func_name](**func_args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc_data["id"],
                "name": func_name,
                "content": json.dumps(result),
            })
        # Continue the conversation loop...
```

---

## Multi-Step Agentic Patterns

Kimi K2.5 supports **interleaved thinking and multi-step tool calling** — the model reasons between tool calls, planning its next action based on previous results. This makes it excellent for complex agentic workflows.

### Complete Agent Example

```python
import json
from openai import OpenAI

client = OpenAI(api_key="your-key", base_url="https://api.moonshot.ai/v1")

tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression.",
            "parameters": {
                "type": "object",
                "required": ["expression"],
                "properties": {
                    "expression": {"type": "string", "description": "e.g. '2+2'"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get current stock price by ticker symbol.",
            "parameters": {
                "type": "object",
                "required": ["ticker"],
                "properties": {
                    "ticker": {"type": "string", "description": "e.g. 'AAPL'"}
                }
            }
        }
    },
]

tool_map = {
    "calculate": lambda expression: {"result": eval(expression)},
    "get_stock_price": lambda ticker: {"ticker": ticker, "price": 150.25, "currency": "USD"},
}

def run_agent(user_query: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful financial assistant with access to tools."},
        {"role": "user", "content": user_query},
    ]

    while True:
        response = client.chat.completions.create(
            model="kimi-k2.5",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.6,
            max_tokens=4096,
        )

        choice = response.choices[0]

        if choice.finish_reason == "tool_calls":
            messages.append(choice.message)
            for tc in choice.message.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments)
                result = tool_map[name](**args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": name,
                    "content": json.dumps(result),
                })
        else:
            return choice.message.content

# The model will: 1) call get_stock_price, 2) call calculate, 3) return final answer
answer = run_agent("What's Apple's stock price? If I buy 100 shares, how much will it cost?")
print(answer)
```

**Capabilities:**
- Stable execution across 200–300+ sequential tool calls
- Autonomous decision on when and which tools to call
- Reasoning between tool calls (with thinking mode enabled)
- Parallel tool calls in a single turn (multiple items in `tool_calls` array)
