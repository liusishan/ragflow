import asyncio
from dataclasses import dataclass
from typing import Any

try:
    from openai import AsyncOpenAI
except Exception:  # pragma: no cover - optional dependency
    AsyncOpenAI = None


@dataclass
class LLMConfig:
    api_key: str
    model: str
    base_url: str | None = None
    max_tokens: int = 8192
    temperature: float = 0.2


class OpenAICompatibleChat:
    def __init__(self, config: LLMConfig):
        if AsyncOpenAI is None:
            raise RuntimeError("openai package is required for OpenAI-compatible chat")
        self.model_name = config.model
        self.max_length = config.max_tokens
        self.client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
        self.temperature = config.temperature

    async def async_chat(self, system: str, history: list[dict], gen_conf: dict | None = None, **kwargs) -> str:
        messages = list(history)
        if system and (not messages or messages[0].get("role") != "system"):
            messages.insert(0, {"role": "system", "content": system})
        conf = {"temperature": self.temperature}
        if gen_conf:
            conf.update(gen_conf)
        response = await self.client.chat.completions.create(model=self.model_name, messages=messages, **conf, **kwargs)
        choice = response.choices[0]
        return (choice.message.content or "").strip()


async def gather_with_concurrency(limit: int, coroutines: list[Any]):
    semaphore = asyncio.Semaphore(limit)

    async def runner(coro):
        async with semaphore:
            return await coro

    return await asyncio.gather(*(runner(coro) for coro in coroutines))
