from fastapi import APIRouter, Depends, UploadFile, File
from ...services.rag_service import RAGService
from typing import Optional
import shutil
import os
from pathlib import Path

router = APIRouter()

def get_rag_service():
    return RAGService()

@router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    rag_service: RAGService = Depends(get_rag_service)
):
    """Upload a document and process it for RAG."""
    # Create uploads directory if it doesn't exist
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    
    # Save the uploaded file
    file_path = upload_dir / file.filename
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Process the document
        result = await rag_service.process_document(str(file_path))
        
        # Clean up the uploaded file
        os.remove(file_path)
        
        return result
    except Exception as e:
        # Clean up on error
        if file_path.exists():
            os.remove(file_path)
        raise

@router.post("/documents/query")
async def query_document(
    query: str,
    rag_service: RAGService = Depends(get_rag_service)
):
    """Query the processed documents."""
    return await rag_service.query_document(query)