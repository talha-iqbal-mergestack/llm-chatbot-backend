from typing import List, Optional
from openai import AsyncOpenAI
from .base import LLMProvider, Message
from ..config import settings

class OpenAIProvider(LLMProvider):
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY, base_url="https://openrouter.ai/api/v1")

    async def chat_completion(
        self,
        messages: List[Message],
        model: Optional[str] = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        formatted_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        response = await self.client.chat.completions.create(
            model=model,
            messages=formatted_messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return response.choices[0].message.content