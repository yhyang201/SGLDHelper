"""OpenAI-compatible client wrapping Kimi K2.5 (Moonshot AI)."""

from __future__ import annotations

import json
from typing import Any

import structlog
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from sgldhelper.config import Settings

log = structlog.get_logger()

# Category list for instant-mode classification
CLASSIFICATION_CATEGORIES = [
    "progress_update",
    "blocker",
    "question",
    "general",
]


class KimiClient:
    """Async wrapper around Kimi K2.5 via OpenAI-compatible API.

    Supports two modes:
    - thinking (default): For complex reasoning + tool calling. temperature=1.0
    - instant: For cheap/fast classification + summaries. temperature=0.6
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = settings.kimi_model
        self._client = AsyncOpenAI(
            api_key=settings.moonshot_api_key,
            base_url=settings.kimi_base_url,
        )

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        thinking: bool = True,
        **kwargs: Any,
    ) -> ChatCompletion:
        """Send a chat completion request.

        Args:
            messages: OpenAI-format message list.
            tools: OpenAI-format tool definitions.
            thinking: True for thinking mode (reasoning), False for instant mode.
            **kwargs: Additional params forwarded to the API.
        """
        extra: dict[str, Any] = {}
        if not thinking:
            extra["extra_body"] = {"thinking": {"type": "disabled"}}

        temperature = 1.0 if thinking else 0.6

        call_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            **extra,
            **kwargs,
        }
        if tools:
            call_kwargs["tools"] = tools

        response = await self._client.chat.completions.create(**call_kwargs)
        return response

    async def classify(
        self,
        text: str,
        categories: list[str] | None = None,
    ) -> dict[str, Any]:
        """Lightweight instant-mode classification.

        Returns dict with 'category' and optionally 'extracted' fields.
        """
        cats = categories or CLASSIFICATION_CATEGORIES
        categories_str = ", ".join(cats)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a message classifier for a software development team. "
                    "Classify the user's message into exactly one category.\n\n"
                    f"Categories: {categories_str}\n\n"
                    "Respond with JSON only:\n"
                    '{"category": "<category>", "summary": "<brief summary>", '
                    '"mentioned_pr": <pr_number_or_null>, "mentioned_feature": "<feature_title_or_null>"}'
                ),
            },
            {"role": "user", "content": text},
        ]

        response = await self.chat(messages, thinking=False)
        content = response.choices[0].message.content or "{}"

        # Parse JSON from response, stripping markdown fences if present
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            log.warning("classifier.json_parse_failed", raw=content[:200])
            return {"category": "general", "summary": content[:100]}

    def extract_usage(self, response: ChatCompletion) -> tuple[int, int]:
        """Extract (tokens_in, tokens_out) from a completion response."""
        usage = response.usage
        if usage:
            return usage.prompt_tokens, usage.completion_tokens
        return 0, 0
