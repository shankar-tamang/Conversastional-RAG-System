import os
import json
import time
import logging
import numpy as np
import asyncio
import tempfile
from uuid import uuid4

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

from langchain_community.document_loaders import PyPDFLoader

from src.chunkers.fixed_size import FixedSizeChunker
from chunkers.token_chunking import SentenceSplitChunker
from src.pinecone_client import PineconeClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
QA_PAIRS_PATH = "tests/test_data/qa_pairs.json"
PDF_PATHS = ["tests/test_data/1301.3781v3.pdf", "tests/test_data/1706.03762v7.pdf"]
CHUNKING_STRATEGIES = ["fixed_size", "sentence_split"]
SIMILARITY_THRESHOLD = 0.80
TOP_K = 5 # Number of chunks to retrieve

# Use the same embedding model as the ingestion service
embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")


class MockDocumentService:
    """A mock DocumentService that bypasses the database and operates synchronously."""
    def __init__(self, namespace):
        self.pinecone_client = PineconeClient(index_name="llama-text-embed-v2-index", namespace=namespace)
        self.embeddings = embedding_model

    def process_document(self, filename: str, content: bytes, chunking_strategy: str):
        documents = self._extract_documents(content, filename)
        
        if chunking_strategy == "fixed_size":
            chunker = FixedSizeChunker(chunk_size=1000, chunk_overlap=200)
        elif chunking_strategy == "sentence_split":
            chunker = SentenceSplitChunker()
        else:
            raise ValueError(f"Unknown chunking strategy: {chunking_strategy}")

        chunks = chunker.chunk(documents)

        records = []
        for doc in chunks:
            records.append({
                "id": str(uuid4()),
                "text": doc.page_content,
            })

        self.pinecone_client._upsert(documents=records)
        # No database interaction
        return True

    def _extract_documents(self, content: bytes, filename: str):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(content)
            temp_pdf_path = temp_pdf.name

        try:
            loader = PyPDFLoader(temp_pdf_path)
            documents = loader.load()
        finally:
            os.remove(temp_pdf_path)
        return documents

def load_qa_pairs():
    """Loads the question-answer pairs from the JSON file."""
    with open(QA_PAIRS_PATH, 'r') as f:
        return json.load(f)

def run_benchmark():
    """Runs the full benchmark for all chunking strategies."""
    qa_pairs = load_qa_pairs()
    overall_results = {}

    for strategy in CHUNKING_STRATEGIES:
        namespace = f"benchmark-{strategy}"
        logging.info(f"--- Starting benchmark for strategy: {strategy} in namespace: {namespace} ---")

        # 1. Ingestion Phase
        logging.info("Clearing old data and starting ingestion...")
        doc_service = MockDocumentService(namespace=namespace)
        # For now, we assume each run is clean or overwrites.

        for pdf_path in PDF_PATHS:
            try:
                with open(pdf_path, "rb") as f:
                    content = f.read()
                filename = os.path.basename(pdf_path)
                doc_service.process_document(filename, content, strategy)
                logging.info(f"Successfully ingested {filename} using {strategy} strategy.")
            except Exception as e:
                logging.error(f"Error ingesting {pdf_path}: {e}")
                continue

        # 2. Evaluation Phase
        logging.info("Starting evaluation phase...")
        metrics = {
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "latencies": []
        }

        for i, item in enumerate(qa_pairs):
            question = item["question"]
            ground_truth_answer = item["answer"]
            logging.info(f"--- Evaluating Q{i+1}/{len(qa_pairs)}: {question} ---")

            start_time = time.time()
            retrieved_chunks = doc_service.pinecone_client.query(question, top_k=TOP_K)
            latency = time.time() - start_time
            metrics["latencies"].append(latency)

            
            if not retrieved_chunks:
                logging.warning("No chunks retrieved for this question.")
                metrics["false_negatives"] += 1
                continue

            retrieved_texts = "\n\n".join([hit['fields']['text'] for hit in retrieved_chunks['result']['hits']])

            Embeddings = embedding_model.embed_documents([ground_truth_answer, retrieved_texts])

            similarities = cosine_similarity([Embeddings[0]], [Embeddings[1]])[0]
            max_similarity = np.max(similarities)
            logging.info(f"Highest similarity score: {max_similarity:.4f}")
            
            if max_similarity >= SIMILARITY_THRESHOLD:
                logging.info(f"SUCCESS: Match found for Q{i+1}.")
                metrics["true_positives"] += 1
            else:
                logging.info(f"FAILURE: No match found for Q{i+1}.")
                metrics["false_negatives"] += 1

        # 3. Calculate Final Metrics
        total_questions = len(qa_pairs)
        accuracy = metrics["true_positives"] / total_questions if total_questions > 0 else 0
        recall = metrics["true_positives"] / (metrics["true_positives"] + metrics["false_negatives"]) if (metrics["true_positives"] + metrics["false_negatives"]) > 0 else 0
        avg_latency = sum(metrics["latencies"]) / len(metrics["latencies"]) if metrics["latencies"] else 0

        overall_results[strategy] = {
            "Accuracy": f"{accuracy:.2%}",
            "Recall": f"{recall:.2%}",
            "Average Latency (s)": f"{avg_latency:.4f}",
            "Total Questions": total_questions,
            "Correctly Answered": metrics["true_positives"]
        }

    # 4. Save and Print Results
    results_path = "benchmarks/chunker_benchmark_results.json"
    with open(results_path, 'w') as f:
        json.dump(overall_results, f, indent=2)
    
    logging.info(f"Benchmark complete. Results saved to {results_path}")
    print(json.dumps(overall_results, indent=2))

if __name__ == "__main__":
    run_benchmark()