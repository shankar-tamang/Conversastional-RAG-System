
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, Form
from sqlalchemy.orm import Session
from src.database import get_db
from src.services.document_service import DocumentService
from typing import Annotated

router = APIRouter()

@router.post("/documents/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    chunking_strategy: Annotated[str, Form()] = "fixed_size",
    db: Session = Depends(get_db),
    namespace: str = Form("example-namespace") 
):
    """
    Upload a document (.pdf or .txt), extract text, chunk it, generate embeddings, 
    and store it in the vector database. Metadata is stored in a SQL database.
    """
    if file.content_type not in ["application/pdf", "text/plain"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF and TXT are supported.")

    try:
        content = await file.read()
        document_service = DocumentService(db, namespace)
        document = await document_service.process_document(file.filename, content, chunking_strategy)
        return {"message": "Document ingested successfully", "document_id": document.id}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

