#!/usr/bin/env python3
import asyncio
import argparse
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredPDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import fitz
import pdfplumber
import uuid
from nltk.tokenize import sent_tokenize
from app.core.config import settings
from app.services.rag_service import RAGService

class PDFProcessor:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            api_key=settings.OPENAI_EMBEDDINGS_KEY,
            base_url="https://api.openai.com/v1"
        )
        self.pinecone_client = Pinecone(
            api_key=settings.PINECONE_API_KEY
        )
        self.index_name = "document-store"
        self.namespace = "unnamed"
    
    def _ensure_index_exists(self):
        existing_indexes = [index.name for index in self.pinecone_client.list_indexes()]
        if self.index_name not in existing_indexes:
            self.pinecone_client.create_index(
                name=self.index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
    
    async def process_document_with_pypdf(self, file_path: str) -> list:
        """Process a document using PyPDFLoader for basic PDF text extraction.
        
        Args:
            file_path (str): Path to the document to process
        
        Returns:
            list: List of document chunks
        """
        
        # Load and process the document
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # Flatten sentences
        sentences = []
        for doc in documents:
            for sentence in sent_tokenize(doc.page_content):
                sentences.append(Document(page_content=sentence))
        
        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=150,
            separators=[
                "\n\n",  # Paragraphs
                "\n",    # Lines
                ".",     # Sentences
                ":",     # Lists/definitions
                "; ",    # Semi-colon separated lists
                ", ",    # Comma-separated items
                " ",     # Words
                ""       # Characters
            ]
        )
        return splitter.split_documents(sentences)

    async def process_document_with_fitz(self, file_path: str) -> list:
        """Process a document using PyMuPDF (fitz) for advanced PDF text extraction.
        
        Args:
            file_path (str): Path to the document to process
        
        Returns:
            list: List of document chunks
        """
        
        # Load and process the document
        doc = fitz.open(file_path)
        documents = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            # Create a Document object with page metadata
            documents.append(Document(
                page_content=text,
                metadata={
                    "source": file_path,
                    "page": page_num + 1,
                    "total_pages": len(doc)
                }
            ))
        
        doc.close()
        
        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=150,
            separators=[
                "\n\n",  # Paragraphs
                "\n",    # Lines
                ".",     # Sentences
                ":",     # Lists/definitions
                "; ",    # Semi-colon separated lists
                ", ",    # Comma-separated items
                " ",     # Words
                ""       # Characters
            ]
        )
        return splitter.split_documents(documents)

    async def process_document_with_pdfplumber(self, file_path: str) -> list:
        """Process a document using pdfplumber for advanced PDF text and table extraction.
        
        Args:
            file_path (str): Path to the document to process
        
        Returns:
            list: List of document chunks
        """
        
        # Load and process the document
        documents = []
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Extract text
                text = page.extract_text()
                
                # Extract tables if any exist
                tables = page.extract_tables()
                if tables:
                    text += "\n\nTables:\n"
                    for table in tables:
                        for row in table:
                            text += " ".join(str(cell) if cell else "" for cell in row) + "\n"
                
                # Create a Document object with page metadata
                documents.append(Document(
                    page_content=text,
                    metadata={
                        "source": file_path,
                        "page": page_num + 1,
                        "total_pages": len(pdf.pages),
                        "has_tables": bool(tables)
                    }
                ))
        
        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=150,
            separators=[
                "\n\n",  # Paragraphs
                "\n",    # Lines
                ".",     # Sentences
                ":",     # Lists/definitions
                "; ",    # Semi-colon separated lists
                ", ",    # Comma-separated items
                " ",     # Words
                ""       # Characters
            ]
        )
        return splitter.split_documents(documents)


    async def process_document(self, file_path: str, mode: str = "fitz") -> dict:
        # Set namespace based on the processing mode
        self.namespace = mode

        # Process document based on mode
        if mode == "pypdf":
            chunks = await self.process_document_with_pypdf(file_path)
        elif mode == "fitz":
            chunks = await self.process_document_with_fitz(file_path)
        elif mode == "pdfplumber":
            chunks = await self.process_document_with_pdfplumber(file_path)
        else:
            raise ValueError(f"Invalid processing mode: {mode}")

        # # Save chunks to a text file for review
        # output_file = f"{file_path}_chunks_{mode}.txt"
        # with open(output_file, 'w', encoding='utf-8') as f:
        #     for i, chunk in enumerate(chunks, 1):
        #         f.write(f"\n=== Chunk {i} ===\n")
        #         f.write(f"Content:\n{chunk.page_content}\n")
        #         f.write(f"Metadata:\n{chunk.metadata}\n")

        # Generate unique IDs for each chunk
        chunk_ids = [f"{mode}_{uuid.uuid4()}" for _ in chunks]


        # Ensure Pinecone index exists and store vectors
        self._ensure_index_exists()
        
        index = self.pinecone_client.Index(self.index_name)
        vectorstore = PineconeVectorStore(
            index=index,
            embedding=self.embeddings,
            text_key="text",
            namespace=self.namespace
        )
        
        vectorstore.add_documents(chunks, ids=chunk_ids)
        
        return {
            "status": "success",
            "chunks": len(chunks),
            "namespace": self.namespace
        }

async def main():
    parser = argparse.ArgumentParser(description='Process PDF documents and store them in Pinecone with library-specific namespaces')
    parser.add_argument('pdf_path', help='Path to the PDF file to process')
    parser.add_argument(
        '--mode',
        choices=['fitz', 'pdfplumber', 'pypdf'],
        default='fitz',
        help='Text extraction mode to use (default: fitz)'
    )

    args = parser.parse_args()
    
    # Initialize processor
    # processor = PDFProcessor()
    processor = RAGService()
    
    try:
        # Process the document
        result = await processor.process_document(
            file_path=args.pdf_path,
            mode=args.mode
        )
        print(f"Successfully processed document:")
        print(f"- Number of chunks: {result['chunks']}")
        print(f"- Namespace: {result['namespace']}")
        print(f"- Status: {result['status']}")
    
    except Exception as e:
        print(f"Error processing document: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())