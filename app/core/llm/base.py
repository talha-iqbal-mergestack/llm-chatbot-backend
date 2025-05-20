from abc import ABC, abstractmethod
from typing import List, Optional

class Message:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

class LLMProvider(ABC):
    @abstractmethod
    async def chat_completion(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate chat completion using the LLM provider."""
        pass