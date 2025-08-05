import os
import json
import time
import logging
import numpy as np
import pinecone
from uuid import uuid4
import tempfile

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.document_loaders import PyPDFLoader

from src.chunkers.fixed_size import FixedSizeChunker

from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
QA_PAIRS_PATH = "tests/test_data/qa_pairs.json"
PDF_PATHS = ["tests/test_data/1301.3781v3.pdf", "tests/test_data/1706.03762v7.pdf"]
CHUNK_STRATEGY = "fixed_size"
SIMILARITY_METRICS = ["dotproduct"]
SIMILARITY_THRESHOLD = 0.70
TOP_K = 5
EMBEDDING_DIMENSION = 768 # For models/text-embedding-004

# Initialize Pinecone connection

embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")


from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from uuid import uuid4
from dotenv import load_dotenv
import os


load_dotenv()

# === Setup Pinecone ===

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

class PineconeClient:
    def __init__(self, index_name: str, namespace: str, metric):
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index_name = index_name
        self.namespace="default"

        self.pc.create_index(
                name=index_name,
                dimension=1024,
                metric=metric,  # Set the metric to dotproduct
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
        self.index = self.pc.Index(self.index_name)
        self.namespace = namespace

    def _upsert(self, documents) :
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






class MockDocumentServiceForSimilarity:
    """A mock DocumentService that uses a specific Pinecone index."""
    def __init__(self, index_name, metric):
        # This client connects to a specific, newly created index
        self.pinecone_client = PineconeClient(index_name=index_name, namespace="default", metric=metric)
        self.chunker = FixedSizeChunker(chunk_size=1000, chunk_overlap=200)

    def process_document(self, filename: str, content: bytes):
        documents = self._extract_documents(content, filename)
        chunks = self.chunker.chunk(documents)
        records = [{"id": str(uuid4()), "text": doc.page_content} for doc in chunks]
        self.pinecone_client._upsert(documents=records)
        return True

    def _extract_documents(self, content: bytes, filename: str):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(content)
            temp_pdf_path = temp_pdf.name
        try:
            return PyPDFLoader(temp_pdf_path).load()
        finally:
            os.remove(temp_pdf_path)

def load_qa_pairs():
    with open(QA_PAIRS_PATH, 'r') as f:
        return json.load(f)

def run_benchmark():
    qa_pairs = load_qa_pairs()
    overall_results = {}

    for metric in SIMILARITY_METRICS:
        index_name = "llama-text-embed-v2-index"
        logging.info(f"--- Starting benchmark for metric: {metric} on index: {index_name} ---")

        try:

            doc_service = MockDocumentServiceForSimilarity(index_name, metric)

            # 2. Ingestion Phase
            logging.info("Starting ingestion with chunk strategy: {CHUNK_STRATEGY}")
            for pdf_path in PDF_PATHS:
                with open(pdf_path, "rb") as f:
                    content = f.read()
                doc_service.process_document(os.path.basename(pdf_path), content)
            logging.info("Ingestion complete.")

            # 3. Evaluation Phase
            logging.info("Starting evaluation...")
            metrics = {"tp": 0, "fp": 0, "fn": 0, "latencies": []}

            for i, item in enumerate(qa_pairs):
                question, ground_truth_answer = item["question"], item["answer"]
                logging.info(f"--- Evaluating Q{i+1}/{len(qa_pairs)}: {question[:50]}... ---")

                start_time = time.time()
                retrieved_chunks_response = doc_service.pinecone_client.query(question, top_k=TOP_K)
                metrics["latencies"].append(time.time() - start_time)

                if not retrieved_chunks_response or not retrieved_chunks_response.get('result', {}).get('hits'):
                    metrics["fn"] += 1
                    continue

                
                retrieved_texts = "\n\n".join([hit['fields']['text'] for hit in retrieved_chunks_response['result']['hits']])

                Embeddings = embedding_model.embed_documents([ground_truth_answer, retrieved_texts])


                similarities = cosine_similarity([Embeddings[0]], [Embeddings[1]])[0]
                
                found_match = False
                for score in similarities:
                    if score >= SIMILARITY_THRESHOLD:
                        found_match = True
                        metrics["tp"] += 1
                    else:
                        metrics["fp"] += 1
                
                if not found_match:
                    metrics["fn"] += 1

            # 4. Calculate Final Metrics
            total_questions = len(qa_pairs)
            tp, fp, fn = metrics["tp"], metrics["fp"], metrics["fn"]
            
            accuracy = tp / total_questions if total_questions > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            avg_latency = sum(metrics["latencies"]) / len(metrics["latencies"]) if metrics["latencies"] else 0

            overall_results[metric] = {
                "Accuracy": f"{accuracy:.2%}",
                "Precision": f"{precision:.2%}",
                "Recall": f"{recall:.2%}",
                "F1-Score": f"{f1_score:.2%}",
                "Average Latency (s)": f"{avg_latency:.4f}"
            }

        finally:
            pass

    # 6. Save and Print Final Report
    results_path = "benchmarks/similarity_benchmark_results.json"
    with open(results_path, 'w') as f:
        json.dump(overall_results, f, indent=2)
    
    logging.info(f"Benchmark complete. Results saved to {results_path}")
    print(json.dumps(overall_results, indent=2))

if __name__ == "__main__":
    run_benchmark()
