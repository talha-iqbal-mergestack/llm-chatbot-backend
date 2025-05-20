from fastapi import APIRouter, Depends
from ...core.llm.base import Message
from ...core.llm.openai_provider import OpenAIProvider
from ...services.chat_service import ChatService
from ...schemas.chat import ChatRequest, ChatResponse

router = APIRouter()

def get_chat_service():
    # In a real application, you might want to use a proper DI container
    # and possibly make the LLM provider configurable
    llm_provider = OpenAIProvider()
    return ChatService(llm_provider)

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service)
) -> ChatResponse:
    """Send a message to the chatbot and get a response."""
    messages = [
        Message(role=msg.role, content=msg.content)
        for msg in request.messages
    ]
    
    response = await chat_service.send_message(
        messages=messages,
        model=request.model,
        temperature=request.temperature,
        max_tokens=request.max_tokens
    )
    
    return ChatResponse(response=response)