from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from uuid import uuid4
from dotenv import load_dotenv
import os


load_dotenv()

# === Setup Pinecone ===

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

class PineconeClient:
    def __init__(self, index_name: str, namespace: str):
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index_name = index_name
        self.index = self.pc.Index(self.index_name)
        self.namespace = namespace

    def _upsert(self, documents: list[dict]) :
        # uuids = [str(uuid4()) for _ in range(len(documents))]
        self.index.upsert_records(
                    namespace=self.namespace,
                    records=documents
                )
        

    def query(self, query, top_k):
        results = self.index.search(
        namespace=self.namespace,
        query={
            "inputs": {"text": query},
            "top_k": top_k
    }
)
        return results




