from langchain_openai import OpenAIEmbeddings
import re
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import PyPDFLoader
import fitz
import pdfplumber
from langchain_community.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain.docstore.document import Document
from nltk.tokenize import sent_tokenize
from ..core.config import settings

class RAGService:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            api_key=settings.OPENAI_EMBEDDINGS_KEY,
            base_url="https://api.openai.com/v1",
        )
        # Initialize the main Pinecone client
        self.pinecone_client = Pinecone(
            api_key=settings.PINECONE_API_KEY
        )
        self.index_name = "document-store"
        self.namespace = ""
    
    def _ensure_index_exists(self):
        # Check if index exists, create if it doesn't
        existing_indexes = [index.name for index in self.pinecone_client.list_indexes()]
        if self.index_name in existing_indexes:
            return True
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
            return False
    
    async def process_single_document(self, file_path: str) -> list:
        """Process a document in single mode - treating the entire document as one continuous text.
        
        Args:
            file_path (str): Path to the document to process
        
        Returns:
            list: List of document chunks
        """
        # Load and process the document
        loader = UnstructuredPDFLoader(file_path, mode="single")
        documents = loader.load()

        # Filter out headers and footers first
        documents = [doc for doc in documents if doc.metadata.get("category") not in {"Header", "Footer"}]

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

    async def process_document_elements(self, file_path: str) -> list:
        """Process a document in elements mode - preserving structural elements.
        
        Args:
            file_path (str): Path to the document to process
        
        Returns:
            list: List of document chunks
        """
        # Load and process the document
        loader = UnstructuredPDFLoader(file_path, mode="elements")
        documents = loader.load()

        # Filter out headers and footers first
        documents = [doc for doc in documents if doc.metadata.get("category") not in {"Header", "Footer"}]

        # Split into chunks while preserving element metadata
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2500,
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

    async def process_documents_with_semantic_grouping(self, file_path: str) -> list:
        """Process a document using semantic grouping - combining related elements before chunking.
        
        Args:
            file_path (str): Path to the document to process
        
        Returns:
            list: List of document chunks
        """
        # Load and process the document
        loader = UnstructuredPDFLoader(file_path, mode="elements")
        documents = loader.load()

        # Filter out headers and footers first
        documents = [doc for doc in documents if doc.metadata.get("category") not in {"Header", "Footer"}]

        combined_docs = []
        buffer = []
        current_title = None

        for i, doc in enumerate(documents):
            category = doc.metadata.get("category")
            
            if category == "Title" or category == "Heading":
                # First, save any existing buffer
                if buffer:
                    metadata = buffer[0].metadata.copy()
                    if current_title:
                        metadata["title"] = current_title
                    combined_docs.append(Document(
                        page_content="\n".join(d.page_content for d in buffer),
                        metadata=metadata
                    ))
                    buffer = []
                # Start new buffer with the title/heading
                current_title = doc.page_content
                buffer = [doc]
            elif category in {"NarrativeText", "ListItem", "Table"}:
                buffer.append(doc)
            else:
                # For any other unknown categories, log them for review
                print(f"Unknown category found: {category}")
                buffer.append(doc)

        # Don't forget the last buffer
        if buffer:
            metadata = buffer[0].metadata.copy()
            if current_title:
                metadata["title"] = current_title
            combined_docs.append(Document(
                page_content="\n".join(d.page_content for d in buffer),
                metadata=metadata
            ))

        # Split into chunks while preserving metadata
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2500,
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

        return splitter.split_documents(combined_docs)

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

    async def process_document(self, file_path: str, mode: str = "pdfplumber") -> dict:
        """Process a document using the specified mode.
        
        Args:
            file_path (str): Path to the document to process
            mode (str): Processing mode - 'single', 'elements', or 'semantic'
        
        Returns:
            dict: Processing status and number of chunks
        """

        # Create namespace based on file name and processing mode
        self.namespace = mode

        # Process document based on mode
        if mode == "single":
            chunks = await self.process_single_document(file_path)
        elif mode == "elements":
            chunks = await self.process_document_elements(file_path)
        elif mode == "semantic":
            chunks = await self.process_documents_with_semantic_grouping(file_path)
        elif mode == "pypdf":
            chunks = await self.process_document_with_pypdf(file_path)
        elif mode == "fitz":
            chunks = await self.process_document_with_fitz(file_path)
        elif mode == "pdfplumber":
            chunks = await self.process_document_with_pdfplumber(file_path)
        else:
            raise ValueError(f"Invalid processing mode: {mode}. Must be 'single', 'elements', or 'semantic'.")

        # Save chunks to a text file for review
        output_file = f"{file_path}_chunks_{mode}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(chunks, 1):
                f.write(f"\n=== Chunk {i} ===\n")
                f.write(f"Content:\n{chunk.page_content}\n")
                f.write(f"Metadata:\n{chunk.metadata}\n")

        # Ensure Pinecone index exists
        self._ensure_index_exists()
        # if (self._ensure_index_exists()):
        #     return
        
        # Store in Pinecone using the dedicated PineconeVectorStore integration
        index = self.pinecone_client.Index(self.index_name)
        vectorstore = PineconeVectorStore(
            index=index,
            embedding=self.embeddings,
            text_key="text",  # Default document text field
            namespace=self.namespace
        )
        
        vectorstore.add_documents(chunks)
        
        return {
            "status": "success", 
            "chunks": len(chunks),
            "namespace":self.namespace
        }
    
    async def query_document(self, query: str) -> dict:
        # Initialize retriever directly from the existing index
        index = self.pinecone_client.Index(self.index_name)
        
        vectorstore = PineconeVectorStore(
            index=index,
            embedding=self.embeddings,
            text_key="text",  # Default document text field
            namespace="pdfplumber"
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
                # api_key=settings.OPENAI_API_KEY,
                # base_url=settings.OPENAI_API_BASE,
                # model="deepseek/deepseek-r1:free",
                api_key=settings.OPENAI_EMBEDDINGS_KEY,
                base_url="https://api.openai.com/v1",
                model="gpt-4.1",
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