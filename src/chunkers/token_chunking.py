
from langchain.text_splitter import TokenTextSplitter

class TokenSplitChunker:
    def __init__(self):
        self.text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=200)

    def chunk(self, text: str) -> list[str]:
     
        return self.text_splitter.split_documents(text)
    




