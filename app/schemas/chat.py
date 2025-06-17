from pydantic import BaseModel, Field
from typing import List, Optional

class MessageSchema(BaseModel):
    role: str = Field(..., description="The role of the message sender (e.g., 'user', 'assistant')")
    content: str = Field(..., description="The content of the message")

class ChatRequest(BaseModel):
    messages: List[MessageSchema] = Field(..., description="List of messages in the conversation")
    model: Optional[str] = Field(None, description="The LLM model to use for the response")
    temperature: float = Field(0.7, description="Temperature for response generation", ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(None, description="Maximum number of tokens in the response")

class ChatResponse(BaseModel):
    response: str = Field(..., description="The response from the LLM")