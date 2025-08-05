import tempfile
from sqlalchemy.orm import Session
from src.models.document import Document
from src.chunkers.fixed_size import FixedSizeChunker
from chunkers.token_chunking import SentenceSplitChunker
from src.pinecone_client import PineconeClient
import os
import io
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from uuid import uuid4

class DocumentService:
    def __init__(self, db: Session, namespace="default"):
        self.db = db
        self.vector_db = PineconeClient(index_name="llama-text-embed-v2-index", namespace=namespace)
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    async def process_document(self, filename: str, content: bytes, chunking_strategy: str):
        documents = self._extract_documents(content, filename)
        
        if chunking_strategy == "fixed_size":
            chunker = FixedSizeChunker(chunk_size=1000, chunk_overlap=200)
        elif chunking_strategy == "sentence_split":
            chunker = SentenceSplitChunker()
        else:
            raise ValueError(f"Unknown chunking strategy: {chunking_strategy}")

        chunks = chunker.chunk(documents)

        # Storing chunks vector using PineconeClient
        records = []
        for doc in chunks:
            content = doc.page_content
            records.append({
                "id": str(uuid4()),
                "text": content,
            }
                
            )

        self.vector_db._upsert(documents=records)

        # Save metadata to SQL DB
        doc = Document(
            filename=filename,
            chunking_strategy=chunking_strategy,
            metadata_={
"num_chunks": len(chunks)}
        )
        self.db.add(doc)
        self.db.commit()
        self.db.refresh(doc)
        return doc

    def _extract_documents(self, content: bytes, filename: str) -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(content)
            temp_pdf_path = temp_pdf.name

        try:
            loader = PyPDFLoader(temp_pdf_path)
            documents = loader.load()
        finally:
            os.remove(temp_pdf_path)

        return documents