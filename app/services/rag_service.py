from langchain_openai import OpenAIEmbeddings
import re
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain.docstore.document import Document
from ..core.config import settings

class RAGService:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_API_BASE,
        )
        # Initialize the main Pinecone client
        self.pinecone_client = Pinecone(
            api_key=settings.PINECONE_API_KEY
        )
        self.index_name = "document-store"
    
    @staticmethod
    def is_header_like(text: str, metadata: dict) -> bool:
        # Common header patterns
        header_patterns = {
            # Phone number patterns (with or without dashes)
            r'\d{2,3}[-]?\d{7,10}',
            # Email-like patterns without @ symbol (often found in headers)
            r'\w+[@]?\w+(?:com|net|org)',
            # Address-like patterns
            r'\d+[-]?[A-Z]\d*,\s*[A-Za-z.,\s]+\d{4,5}',
            # Short text that looks like a header
            r'^.{1,50}$'
        }
        
        # Check if text matches header patterns
        if any(re.search(pattern, text.strip()) for pattern in header_patterns):
            return True
            
        # Headers are usually short
        if len(text.split()) < 5 and metadata.get("category") == "Title":
            return True
            
        return False
    
    def _ensure_index_exists(self):
        # Check if index exists, create if it doesn't
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
    
    async def process_document(self, file_path: str) -> dict:
        # Load and process the document
        loader = UnstructuredPDFLoader(file_path, mode="elements")
        documents = loader.load()

        # Filter out headers and footers first
        documents = [doc for doc in documents if doc.metadata.get("category") not in {"Header", "Footer"}]

        # Additional filtering for header-like content
        # documents = [doc for doc in documents if not self.is_header_like(doc.page_content, doc.metadata)]

        combined_docs = []
        buffer = []

        for i, doc in enumerate(documents):
            category = doc.metadata.get("category")
            
            if category == "Title" or category == "Heading":
                # First, save any existing buffer
                if buffer:
                    combined_docs.append(Document(page_content="\n".join(d.page_content for d in buffer), metadata=buffer[0].metadata))
                    buffer = []
                # Start new buffer with the title/heading
                buffer = [doc]
            elif category in {"NarrativeText", "ListItem", "Table"}:
                buffer.append(doc)
            else:
                # For any other unknown categories, log them for review
                print(f"Unknown category found: {category}")
                buffer.append(doc)

        # Don't forget the last buffer
        if buffer:
            combined_docs.append(Document(page_content="\n".join(d.page_content for d in buffer)))
        
        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=100,
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
        # chunks = splitter.split_documents(documents)
        chunks = splitter.split_documents(combined_docs)

        # Save chunks to a text file for review
        output_file = f"{file_path}_chunks.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(chunks, 1):
                f.write(f"\n=== Chunk {i} ===\n")
                f.write(f"Content:\n{chunk.page_content}\n")
                f.write(f"Metadata:\n{chunk.metadata}\n")
        
        # Ensure Pinecone index exists
        self._ensure_index_exists()
        
        # Store in Pinecone using the dedicated PineconeVectorStore integration
        # Create a Pinecone index object first
        index = self.pinecone_client.Index(self.index_name)
        
        # Use the index with PineconeVectorStore
        vectorstore = PineconeVectorStore(
            index=index,
            embedding=self.embeddings,
            text_key="text"  # Default document text field
        )
        
        # Add documents to the vectorstore
        vectorstore.add_documents(chunks)
        
        return {"status": "success", "chunks": len(chunks)}
    
    async def query_document(self, query: str) -> dict:
        # Initialize retriever directly from the existing index
        index = self.pinecone_client.Index(self.index_name)
        
        vectorstore = PineconeVectorStore(
            index=index,
            embedding=self.embeddings,
            text_key="text"  # Default document text field
        )
        retriever = vectorstore.as_retriever()
        
        # Define HR assistant prompt template
        prompt_template = PromptTemplate(
            template="""
            You are an HR assistant. Answer the question using the provided context.
            If the answer includes a table, preserve its structure using markdown table formatting.

            Context:
            {context}

            Question: {question}
            Answer:
            """,
            input_variables=["context", "question"]
        )
        
        # Create QA chain with custom prompt
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(
                api_key=settings.OPENAI_API_KEY,
                base_url=settings.OPENAI_API_BASE,
                model="gpt-3.5-turbo",
            ),
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt_template},
            retriever=retriever,
            return_source_documents=True
        )
        
        # Get response
        response = qa_chain({"query": query})
        
        return {
            "answer": response["result"],
            "sources": [
                {
                    "metadata": doc.metadata,
                    "content": doc.page_content[:300]
                }
                for doc in response["source_documents"]
            ]
        }