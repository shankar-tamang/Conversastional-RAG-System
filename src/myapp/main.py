
from fastapi import FastAPI
from src.api import document_ingest, chat_rag, booking
from src.database import engine, Base
from src.pinecone_client import PineconeClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os

# Create all database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="PalmMind API",
    description="APIs for document ingestion and conversational RAG.",
    version="0.1.0",
)

# @app.on_event("startup")
# async def startup_event():
#     print("Initializing clients...")
#     # Initialize GoogleGenerativeAIEmbeddings
#     app.state.embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
#     # Initialize PineconeClient
#     app.state.pinecone_client = PineconeClient(index_name="index1", namespace="default")
#     print("Clients initialized.")

app.include_router(document_ingest.router, prefix="/api/v1", tags=["Document Ingestion"])
app.include_router(chat_rag.router, prefix="/api/v1", tags=["Conversational RAG"])
app.include_router(booking.router, prefix="/api/v1", tags=["Booking"])

@app.get("/")
async def root():
    return {"message": "Welcome to PalmMind API"}

