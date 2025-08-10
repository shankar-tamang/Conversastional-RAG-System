from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from dotenv import load_dotenv
import getpass
from uuid import uuid4



load_dotenv()

# === Setup Pinecone ===


if not os.getenv("PINECONE_API_KEY"):
    os.environ["PINECONE_API_KEY"] = getpass.getpass("Enter your Pinecone API key: ")

GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
 
# === Initialization ===


index_name = "langchain-test-index"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)

# === Embedding Model === 

embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# === Vector Store Initialization ===

vector_store = PineconeVectorStore(index=index, embedding=embedding_model)

results = vector_store.similarity_search(
    "What is transformer?",
    k=3,
)

print(type(results))

print("###############")

for res in results:
    print(f"* {res.page_content}]")

