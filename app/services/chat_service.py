from typing import List, Optional
from ..core.llm.base import LLMProvider, Message

class ChatService:
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider

    async def send_message(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """Send a message to the LLM and get the response."""
        try:
            response = await self.llm_provider.chat_completion(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response
        except Exception as e:
            # In a production environment, you'd want to log this error
            raise Exception(f"Error getting chat completion: {str(e)}")