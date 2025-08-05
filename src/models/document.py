
from sqlalchemy import Column, String, Integer, JSON
from src.database import Base

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    chunking_strategy = Column(String)
    metadata_ = Column("metadata", JSON)
