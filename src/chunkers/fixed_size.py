
from langchain.text_splitter import RecursiveCharacterTextSplitter

class FixedSizeChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def chunk(self, documents) :
        return self.text_splitter.split_documents(documents)
